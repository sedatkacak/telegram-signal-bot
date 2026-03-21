import logging
import math
import os
from typing import Dict, List, Tuple

import ccxt
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
EXCHANGE_ID = os.getenv('EXCHANGE_ID', 'bybit')
SYMBOLS = [s.strip() for s in os.getenv('SYMBOLS', 'BTC/USDT,ETH/USDT,SOL/USDT,LTC/USDT').split(',') if s.strip()]

TIMEFRAMES = {
    '4h': {'label': '4 Saatlik', 'limit': 260},
    '1d': {'label': '1 Günlük', 'limit': 260},
}


def build_exchange():
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({'enableRateLimit': True})
    exchange.load_markets()
    return exchange


exchange = build_exchange()


def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < 2:
        return [50.0] * len(values)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[1: period + 1]) / period if len(gains) > period else 0.0
    avg_loss = sum(losses[1: period + 1]) / period if len(losses) > period else 0.0
    rsis = [50.0] * len(values)

    for i in range(period + 1, len(values)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        if avg_loss == 0:
            rsis[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsis[i] = 100 - (100 / (1 + rs))
    return rsis



def fetch_ohlcv(symbol: str, timeframe: str, limit: int):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)



def analyze_symbol(symbol: str, timeframe: str) -> Dict[str, float | str]:
    cfg = TIMEFRAMES[timeframe]
    rows = fetch_ohlcv(symbol, timeframe, cfg['limit'])
    if len(rows) < 210:
        raise ValueError(f'{symbol} için yeterli veri yok')

    closes = [float(r[4]) for r in rows]
    highs = [float(r[2]) for r in rows]
    lows = [float(r[3]) for r in rows]
    volumes = [float(r[5]) for r in rows]

    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)
    rsi14 = rsi(closes, 14)

    price = closes[-1]
    prev_price = closes[-2]
    vol_avg20 = sum(volumes[-21:-1]) / 20
    recent_high = max(highs[-21:-1])
    recent_low = min(lows[-21:-1])

    trend_up = ema50[-1] > ema200[-1] and price > ema50[-1]
    trend_down = ema50[-1] < ema200[-1] and price < ema50[-1]
    breakout_up = price > recent_high and volumes[-1] > vol_avg20
    breakout_down = price < recent_low and volumes[-1] > vol_avg20
    rsi_now = rsi14[-1]

    signal = 'BEKLE'
    entry_low = price * 0.995
    entry_high = price * 1.005
    stop = price * 0.98
    tp1 = price * 1.02
    tp2 = price * 1.04
    reason = 'Net kurulum yok.'

    if trend_up and rsi_now >= 52 and breakout_up:
        signal = 'AL'
        entry_low = min(price, recent_high) * 0.997
        entry_high = price * 1.003
        stop = min(recent_low, price * 0.985)
        tp1 = price * 1.02
        tp2 = price * 1.04
        reason = 'Trend yukarı, RSI güçlü ve hacimli kırılım var.'
    elif trend_down and rsi_now <= 48 and breakout_down:
        signal = 'SAT'
        entry_low = price * 0.997
        entry_high = max(price, recent_low) * 1.003
        stop = max(recent_high, price * 1.015)
        tp1 = price * 0.98
        tp2 = price * 0.96
        reason = 'Trend aşağı, RSI zayıf ve hacimli kırılım var.'
    else:
        if trend_up:
            reason = 'Trend yukarı ama giriş için daha temiz kırılım beklenmeli.'
        elif trend_down:
            reason = 'Trend aşağı ama satış için daha temiz kırılım beklenmeli.'

    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'price': price,
        'signal': signal,
        'entry_low': entry_low,
        'entry_high': entry_high,
        'stop': stop,
        'tp1': tp1,
        'tp2': tp2,
        'reason': reason,
        'rsi': rsi_now,
        'ema50': ema50[-1],
        'ema200': ema200[-1],
        'prev_price': prev_price,
    }



def fmt_num(x: float) -> str:
    if x >= 1000:
        return f'{x:,.2f}'
    if x >= 1:
        return f'{x:.4f}'
    return f'{x:.6f}'



def build_message(symbol: str) -> str:
    a4 = analyze_symbol(symbol, '4h')
    a1 = analyze_symbol(symbol, '1d')

    return (
        f'*{symbol}*\n'
        f'Anlık fiyat: `{fmt_num(a4["price"])}`\n\n'
        f'*4 Saatlik*\n'
        f'Sinyal: *{a4["signal"]}*\n'
        f'Alım / işlem bölgesi: `{fmt_num(a4["entry_low"])} - {fmt_num(a4["entry_high"])} `\n'
        f'Stop: `{fmt_num(a4["stop"] )}`\n'
        f'Hedef 1: `{fmt_num(a4["tp1"] )}`\n'
        f'Hedef 2: `{fmt_num(a4["tp2"] )}`\n'
        f'Not: {a4["reason"]}\n\n'
        f'*1 Günlük*\n'
        f'Sinyal: *{a1["signal"]}*\n'
        f'Alım / işlem bölgesi: `{fmt_num(a1["entry_low"])} - {fmt_num(a1["entry_high"])} `\n'
        f'Stop: `{fmt_num(a1["stop"] )}`\n'
        f'Hedef 1: `{fmt_num(a1["tp1"] )}`\n'
        f'Hedef 2: `{fmt_num(a1["tp2"] )}`\n'
        f'Not: {a1["reason"]}'
    )



def symbol_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(symbol, callback_data=f'show:{symbol}')] for symbol in SYMBOLS]
    return InlineKeyboardMarkup(rows)



def detail_keyboard(symbol: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton('🔄 Güncelle', callback_data=f'show:{symbol}')],
        [InlineKeyboardButton('⬅️ Listeye Dön', callback_data='menu')],
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = 'Coin seç. Sana 4 saatlik ve 1 günlük görünüm göstereyim.'
    if update.message:
        await update.message.reply_text(text, reply_markup=symbol_keyboard())
    else:
        await update.effective_chat.send_message(text, reply_markup=symbol_keyboard())


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == 'menu':
        await query.edit_message_text('Coin seç:', reply_markup=symbol_keyboard())
        return

    if query.data and query.data.startswith('show:'):
        symbol = query.data.split(':', 1)[1]
        try:
            text = build_message(symbol)
        except Exception as e:
            logger.exception('analysis failed')
            text = f'Bir hata oluştu: {e}\n\nTekrar deneyebilirsin.'
        await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=detail_keyboard(symbol))


async def health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Bot çalışıyor.')



def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError('TELEGRAM_BOT_TOKEN eksik')
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('health', health))
    app.add_handler(CallbackQueryHandler(on_button))
    logger.info('Bot başlatılıyor...')
    app.run_polling(drop_pending_updates=True)


if __name__ == '__main__':
    main()
