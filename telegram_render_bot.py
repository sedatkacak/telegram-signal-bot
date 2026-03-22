import os
import math
import logging
from typing import Dict, List, Tuple

import ccxt
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

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


def analyze_symbol(symbol: str, timeframe: str, limit: int) -> Dict:
    candles = fetch_ohlcv(symbol, timeframe, limit)
    if len(candles) < 220:
        raise ValueError(f"{symbol} için yeterli veri yok")

    opens = [c[1] for c in candles]
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

    bullish_trend = price > ema50 and ema50 > ema200
    bearish_trend = price < ema50 and ema50 < ema200
    bullish_momentum = rsi14 >= 55 and macd_now > macd_sig_now
    bearish_momentum = rsi14 <= 45 and macd_now < macd_sig_now
    strong_volume = not math.isnan(vol_sma20) and vol_sma20 > 0 and volumes[-1] > vol_sma20
    breakout_up = price >= recent_high * 0.995
    breakdown_down = price <= recent_low * 1.005

    long_entry_low = ema50 * 0.995
    long_entry_high = ema50 * 1.005
    long_stop = min(recent_low, ema50 * 0.985)
    long_tp1 = price * 1.02
    long_tp2 = price * 1.04

    short_entry_low = ema50 * 0.995
    short_entry_high = ema50 * 1.005
    short_stop = max(recent_high, ema50 * 1.015)
    short_tp1 = price * 0.98
    short_tp2 = price * 0.96

    long_entry = (long_entry_low + long_entry_high) / 2
    short_entry = (short_entry_low + short_entry_high) / 2

    long_risk = max(long_entry - long_stop, 0.0)
    long_reward = max(long_tp1 - long_entry, 0.0)
    long_rr = round(long_reward / long_risk, 2) if long_risk > 0 else 0.0

    short_risk = max(short_stop - short_entry, 0.0)
    short_reward = max(short_entry - short_tp1, 0.0)
    short_rr = round(short_reward / short_risk, 2) if short_risk > 0 else 0.0

    long_score = 0
    short_score = 0
    notes = []

    if bullish_trend:
        long_score += 1
        notes.append("Trend yukarı")
    elif bearish_trend:
        short_score += 1
        notes.append("Trend aşağı")
    else:
        notes.append("Trend karışık")

    if bullish_momentum:
        long_score += 1
        notes.append("Momentum alıcı")
    elif bearish_momentum:
        short_score += 1
        notes.append("Momentum satıcı")
    else:
        notes.append("Momentum zayıf")

    if strong_volume:
        if bullish_trend:
            long_score += 1
        elif bearish_trend:
            short_score += 1
        notes.append("Hacim güçlü")
    else:
        notes.append("Hacim zayıf")

    if breakout_up and bullish_trend:
        long_score += 1
        notes.append("Yukarı kırılım yakın")
    if breakdown_down and bearish_trend:
        short_score += 1
        notes.append("Aşağı kırılım yakın")

    if long_rr >= 2.0:
        long_score += 1
    if short_rr >= 2.0:
        short_score += 1

    if long_score >= 4 and bullish_trend and bullish_momentum and long_rr >= 2.0:
        signal = "GÜÇLÜ AL"
        entry_low, entry_high = long_entry_low, long_entry_high
        stop, tp1, tp2 = long_stop, long_tp1, long_tp2
        rr = long_rr
        score = long_score
        note = " | ".join(notes)
    elif short_score >= 4 and bearish_trend and bearish_momentum and short_rr >= 2.0:
        signal = "GÜÇLÜ SAT"
        entry_low, entry_high = short_entry_low, short_entry_high
        stop, tp1, tp2 = short_stop, short_tp1, short_tp2
        rr = short_rr
        score = short_score
        note = " | ".join(notes)
    else:
        signal = "BEKLE"
        if bullish_trend:
            entry_low, entry_high = long_entry_low, long_entry_high
            stop, tp1, tp2 = long_stop, long_tp1, long_tp2
            rr = long_rr
            score = long_score
            note = "Kurulum var ama yeterince güçlü değil."
        elif bearish_trend:
            entry_low, entry_high = short_entry_low, short_entry_high
            stop, tp1, tp2 = short_stop, short_tp1, short_tp2
            rr = short_rr
            score = short_score
            note = "Kurulum var ama yeterince güçlü değil."
        else:
            entry_low, entry_high = price * 0.995, price * 1.005
            stop, tp1, tp2 = price * 0.98, price * 1.02, price * 1.04
            rr = 0.0
            score = 1
            note = "Trend karışık, net işlem yok."

    return {
        "symbol": symbol,
        "price": price,
        "signal": signal,
        "score": min(score, 5),
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "note": note,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi14,
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
    direction_note = ""
    if data_4h["signal"] == data_1d["signal"] and data_4h["signal"] in ("GÜÇLÜ AL", "GÜÇLÜ SAT"):
        direction_note = "Çoklu zaman dilimi uyumlu."
    elif data_4h["signal"] in ("GÜÇLÜ AL", "GÜÇLÜ SAT") and data_1d["signal"] == "BEKLE":
        direction_note = "Kısa vadede sinyal var ama günlük teyit sınırlı."
    elif data_4h["signal"] == "BEKLE" and data_1d["signal"] in ("GÜÇLÜ AL", "GÜÇLÜ SAT"):
        direction_note = "Günlük yön var ama kısa vadeli giriş net değil."
    else:
        direction_note = "Net kurulum yok."

    text = f"""
*{symbol}*
Anlık fiyat: `{fmt_price(data_4h["price"])}`

*4 Saatlik*
Sinyal: *{data_4h["signal"]}*
Güven skoru: *{data_4h["score"]}/5*
İşlem bölgesi: `{fmt_price(data_4h["entry_low"])}` - `{fmt_price(data_4h["entry_high"])}`
Stop: `{fmt_price(data_4h["stop"])}`
Hedef 1: `{fmt_price(data_4h["tp1"])}`
Hedef 2: `{fmt_price(data_4h["tp2"])}`
Risk/Ödül: `{data_4h["rr"]}`
Not: {data_4h["note"]}

*1 Günlük*
Sinyal: *{data_1d["signal"]}*
Güven skoru: *{data_1d["score"]}/5*
İşlem bölgesi: `{fmt_price(data_1d["entry_low"])}` - `{fmt_price(data_1d["entry_high"])}`
Stop: `{fmt_price(data_1d["stop"])}`
Hedef 1: `{fmt_price(data_1d["tp1"])}`
Hedef 2: `{fmt_price(data_1d["tp2"])}`
Risk/Ödül: `{data_1d["rr"]}`
Not: {data_1d["note"]}

*Genel Yorum:* {direction_note}
"""
    return text


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
