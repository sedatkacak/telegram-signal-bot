import os
import math
import logging
from typing import Dict, List, Tuple

import ccxt
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kucoin")
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,LTC/USDT").split(",")]

TIMEFRAMES = {
    "4h": {"label": "4 Saatlik", "limit": 260},
    "1d": {"label": "1 Günlük", "limit": 260},
}


def build_exchange():
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


exchange = build_exchange()


def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    result = [values[0]]
    for price in values[1:]:
        result.append(price * k + result[-1] * (1 - k))
    return result


def sma(values: List[float], period: int) -> List[float]:
    out = []
    for i in range(len(values)):
        if i + 1 < period:
            out.append(float("nan"))
        else:
            window = values[i + 1 - period:i + 1]
            out.append(sum(window) / period)
    return out


def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return [50.0] * len(values)

    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period
    rsis = [50.0] * len(values)

    if avg_loss == 0:
        rsis[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsis[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, len(values)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        if avg_loss == 0:
            rsis[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsis[i] = 100 - (100 / (1 + rs))

    return rsis


def macd(values: List[float]) -> Tuple[List[float], List[float], List[float]]:
    ema12 = ema(values, 12)
    ema26 = ema(values, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal_line = ema(macd_line, 9)
    hist = [a - b for a, b in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist


def fetch_ohlcv(symbol: str, timeframe: str, limit: int):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:,.6f}"


def classify_trade_type(timeframe: str, rr: float, score: int) -> str:
    if timeframe == "4h":
        if score >= 4 and rr >= 2.0:
            return "Swing"
        return "Scalp / Kısa vade"
    return "Swing / Orta vade"


def classify_entry_quality(price: float, entry_low: float, entry_high: float) -> str:
    center = (entry_low + entry_high) / 2
    band = max((entry_high - entry_low) / 2, 1e-9)
    distance = abs(price - center) / band

    if distance <= 0.4:
        return "İyi"
    if distance <= 1.0:
        return "Orta"
    return "Geç"


def score_market(
    price: float,
    ema50: float,
    ema200: float,
    rsi14: float,
    macd_now: float,
    macd_sig_now: float,
    volume_now: float,
    volume_avg: float,
    recent_high: float,
    recent_low: float,
    rr_long: float,
    rr_short: float,
) -> Dict:
    bullish_trend = price > ema50 and ema50 > ema200
    bearish_trend = price < ema50 and ema50 < ema200

    above_ema50 = price > ema50
    below_ema50 = price < ema50

    bullish_momentum = rsi14 >= 54 and macd_now > macd_sig_now
    bearish_momentum = rsi14 <= 46 and macd_now < macd_sig_now

    strong_volume = volume_avg > 0 and volume_now > volume_avg * 1.05

    breakout_up = price >= recent_high * 0.995
    breakdown_down = price <= recent_low * 1.005

    long_score = 0
    short_score = 0
    long_notes = []
    short_notes = []

    if bullish_trend:
        long_score += 2
        long_notes.append("Trend yukarı")
    elif above_ema50:
        long_score += 1
        long_notes.append("Fiyat EMA50 üstü")

    if bearish_trend:
        short_score += 2
        short_notes.append("Trend aşağı")
    elif below_ema50:
        short_score += 1
        short_notes.append("Fiyat EMA50 altı")

    if bullish_momentum:
        long_score += 1
        long_notes.append("Momentum pozitif")
    if bearish_momentum:
        short_score += 1
        short_notes.append("Momentum negatif")

    if strong_volume:
        if bullish_trend or above_ema50:
            long_score += 1
            long_notes.append("Hacim destekli")
        if bearish_trend or below_ema50:
            short_score += 1
            short_notes.append("Hacim destekli")

    if breakout_up:
        long_score += 1
        long_notes.append("Yukarı kırılım yakın")
    if breakdown_down:
        short_score += 1
        short_notes.append("Aşağı kırılım yakın")

    if rr_long >= 2.0:
        long_score += 1
        long_notes.append(f"RR iyi ({rr_long})")
    elif rr_long >= 1.4:
        long_score += 0.5
        long_notes.append(f"RR kabul edilebilir ({rr_long})")

    if rr_short >= 2.0:
        short_score += 1
        short_notes.append(f"RR iyi ({rr_short})")
    elif rr_short >= 1.4:
        short_score += 0.5
        short_notes.append(f"RR kabul edilebilir ({rr_short})")

    return {
        "long_score": long_score,
        "short_score": short_score,
        "long_notes": long_notes,
        "short_notes": short_notes,
        "bullish_trend": bullish_trend,
        "bearish_trend": bearish_trend,
    }


def signal_from_score(score: float, direction: str) -> str:
    if score >= 4.5:
        return "GÜÇLÜ AL" if direction == "long" else "GÜÇLÜ SAT"
    if score >= 3.0:
        return "AL" if direction == "long" else "SAT"
    return "BEKLE"


def analyze_symbol(symbol: str, timeframe: str, limit: int) -> Dict:
    candles = fetch_ohlcv(symbol, timeframe, limit)
    if len(candles) < 220:
        raise ValueError(f"{symbol} için yeterli veri yok")

    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    volumes = [c[5] for c in candles]

    price = closes[-1]
    ema50 = ema(closes, 50)[-1]
    ema200 = ema(closes, 200)[-1]
    rsi14 = rsi(closes, 14)[-1]
    macd_line, macd_signal, _ = macd(closes)
    macd_now = macd_line[-1]
    macd_sig_now = macd_signal[-1]

    vol_sma20 = sma(volumes, 20)[-1]
    recent_high = max(highs[-20:])
    recent_low = min(lows[-20:])

    # Long setup
    long_entry_low = ema50 * 0.995
    long_entry_high = ema50 * 1.005
    long_stop = min(recent_low, ema50 * 0.985)
    long_tp1 = price * 1.02
    long_tp2 = price * 1.04
    long_entry = (long_entry_low + long_entry_high) / 2
    long_risk = max(long_entry - long_stop, 0.0)
    long_reward = max(long_tp1 - long_entry, 0.0)
    long_rr = round(long_reward / long_risk, 2) if long_risk > 0 else 0.0

    # Short setup
    short_entry_low = ema50 * 0.995
    short_entry_high = ema50 * 1.005
    short_stop = max(recent_high, ema50 * 1.015)
    short_tp1 = price * 0.98
    short_tp2 = price * 0.96
    short_entry = (short_entry_low + short_entry_high) / 2
    short_risk = max(short_stop - short_entry, 0.0)
    short_reward = max(short_entry - short_tp1, 0.0)
    short_rr = round(short_reward / short_risk, 2) if short_risk > 0 else 0.0

    scored = score_market(
        price=price,
        ema50=ema50,
        ema200=ema200,
        rsi14=rsi14,
        macd_now=macd_now,
        macd_sig_now=macd_sig_now,
        volume_now=volumes[-1],
        volume_avg=vol_sma20 if not math.isnan(vol_sma20) else 0.0,
        recent_high=recent_high,
        recent_low=recent_low,
        rr_long=long_rr,
        rr_short=short_rr,
    )

    long_signal = signal_from_score(scored["long_score"], "long")
    short_signal = signal_from_score(scored["short_score"], "short")

    # Hangisi daha kuvvetli?
    if scored["long_score"] > scored["short_score"] and long_signal != "BEKLE":
        signal = long_signal
        score = min(int(round(scored["long_score"])), 5)
        entry_low, entry_high = long_entry_low, long_entry_high
        stop, tp1, tp2 = long_stop, long_tp1, long_tp2
        rr = long_rr
        note = " | ".join(scored["long_notes"]) if scored["long_notes"] else "Long taraf daha güçlü."
    elif scored["short_score"] > scored["long_score"] and short_signal != "BEKLE":
        signal = short_signal
        score = min(int(round(scored["short_score"])), 5)
        entry_low, entry_high = short_entry_low, short_entry_high
        stop, tp1, tp2 = short_stop, short_tp1, short_tp2
        rr = short_rr
        note = " | ".join(scored["short_notes"]) if scored["short_notes"] else "Short taraf daha güçlü."
    else:
        signal = "BEKLE"
        score = max(min(int(round(max(scored["long_score"], scored["short_score"]))), 5), 1)
        if scored["bullish_trend"]:
            entry_low, entry_high = long_entry_low, long_entry_high
            stop, tp1, tp2 = long_stop, long_tp1, long_tp2
            rr = long_rr
            note = "Long taraf var ama yeterince güçlü değil."
        elif scored["bearish_trend"]:
            entry_low, entry_high = short_entry_low, short_entry_high
            stop, tp1, tp2 = short_stop, short_tp1, short_tp2
            rr = short_rr
            note = "Short taraf var ama yeterince güçlü değil."
        else:
            entry_low, entry_high = price * 0.995, price * 1.005
            stop, tp1, tp2 = price * 0.98, price * 1.02, price * 1.04
            rr = 0.0
            note = "Trend karışık, net işlem yok."

    trade_type = classify_trade_type(timeframe, rr, score)
    entry_quality = classify_entry_quality(price, entry_low, entry_high)

    return {
        "symbol": symbol,
        "price": price,
        "signal": signal,
        "score": score,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "note": note,
        "trade_type": trade_type,
        "entry_quality": entry_quality,
    }


def build_menu() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(symbol, callback_data=f"symbol:{symbol}")] for symbol in SYMBOLS]
    return InlineKeyboardMarkup(rows)


def build_detail_buttons(symbol: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Güncelle", callback_data=f"refresh:{symbol}")],
        [InlineKeyboardButton("↩️ Listeye Dön", callback_data="back:list")],
    ])


def build_message(symbol: str, data_4h: Dict, data_1d: Dict) -> str:
    if data_4h["signal"] in ("GÜÇLÜ AL", "AL") and data_1d["signal"] in ("GÜÇLÜ AL", "AL"):
        general = "Yön yukarı tarafı destekliyor. Kademeli giriş düşünülebilir."
    elif data_4h["signal"] in ("GÜÇLÜ SAT", "SAT") and data_1d["signal"] in ("GÜÇLÜ SAT", "SAT"):
        general = "Yön aşağı tarafı destekliyor. Zayıflık sürerse satış baskısı devam edebilir."
    elif data_4h["signal"] in ("GÜÇLÜ AL", "AL") and data_1d["signal"] == "BEKLE":
        general = "Kısa vadeli long var ama günlük teyit zayıf."
    elif data_4h["signal"] in ("GÜÇLÜ SAT", "SAT") and data_1d["signal"] == "BEKLE":
        general = "Kısa vadeli short baskısı var ama günlük teyit zayıf."
    else:
        general = "Net kurulum yok."

    return f"""
*{symbol}*
Anlık fiyat: `{fmt_price(data_4h["price"])}`

*4 Saatlik*
Sinyal: *{data_4h["signal"]}*
Güven skoru: *{data_4h["score"]}/5*
İşlem tipi: *{data_4h["trade_type"]}*
Giriş kalitesi: *{data_4h["entry_quality"]}*
İşlem bölgesi: `{fmt_price(data_4h["entry_low"])}` - `{fmt_price(data_4h["entry_high"])}`
Stop: `{fmt_price(data_4h["stop"])}`
Hedef 1: `{fmt_price(data_4h["tp1"])}`
Hedef 2: `{fmt_price(data_4h["tp2"])}`
Risk/Ödül: `{data_4h["rr"]}`
Not: {data_4h["note"]}

*1 Günlük*
Sinyal: *{data_1d["signal"]}*
Güven skoru: *{data_1d["score"]}/5*
İşlem tipi: *{data_1d["trade_type"]}*
Giriş kalitesi: *{data_1d["entry_quality"]}*
İşlem bölgesi: `{fmt_price(data_1d["entry_low"])}` - `{fmt_price(data_1d["entry_high"])}`
Stop: `{fmt_price(data_1d["stop"])}`
Hedef 1: `{fmt_price(data_1d["tp1"])}`
Hedef 2: `{fmt_price(data_1d["tp2"])}`
Risk/Ödül: `{data_1d["rr"]}`
Not: {data_1d["note"]}

*Genel Yorum:* {general}
"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("Coin seç:", reply_markup=build_menu())


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return

    await query.answer()
    data = query.data or ""

    if data == "back:list":
        await query.edit_message_text("Coin seç:", reply_markup=build_menu())
        return

    if data.startswith("symbol:"):
        symbol = data.split(":", 1)[1]
    elif data.startswith("refresh:"):
        symbol = data.split(":", 1)[1]
    else:
        return

    try:
        data_4h = analyze_symbol(symbol, "4h", TIMEFRAMES["4h"]["limit"])
        data_1d = analyze_symbol(symbol, "1d", TIMEFRAMES["1d"]["limit"])
        text = build_message(symbol, data_4h, data_1d)

        await query.edit_message_text(
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=build_detail_buttons(symbol),
        )
    except Exception as e:
        logger.exception("Callback işleminde hata")
        await query.edit_message_text(
            text=f"Bir hata oluştu: {e}\n\nTekrar /start yazıp yeniden deneyebilirsin.",
            reply_markup=build_menu(),
        )


def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN eksik")

    logger.info("Bot başlatılıyor...")
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
