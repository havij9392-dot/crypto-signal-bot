"""
Phase 3 - Signal engine.

Scans the market for the one setup that survived out-of-sample testing:
an EMA20/EMA50 cross, filtered by the 200 EMA on the same timeframe and by
Bitcoin's own trend. Everything else was rejected by the backtest and is not
wired in here.

Runs are idempotent: a signal is sent once per candle close, tracked in
state/last_signals.json, so running every 15 minutes does not spam you.

Usage:
    python scanner.py           # scan and print
    python scanner.py --send    # scan and push to Telegram
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from backtest import add_indicators
from data import fetch_klines, top_symbols
from notify import send

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, "state")
STATE_FILE = os.path.join(STATE_DIR, "last_signals.json")

INTERVAL = "30m"            # entry timeframe
LOOKBACK_DAYS = 30          # ~1440 candles, plenty for a 200 EMA on 30m

# The market regime is read from 4h, not 30m: a 200 EMA on 30m only covers four
# days and flips several times a week, which turns the filter into noise.
REGIME_INTERVAL = "4h"
REGIME_DAYS = 90

# The exact configuration chosen on in-sample data in research.py.
SL_ATR = 2.5
RR = 3.0

PAPER_BANNER = "⚠️ حالت آزمایشی — این سیگنال هنوز با پول واقعی تأیید نشده"


def candles_to_frame(raw: list[list]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open_time": [
                datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                )
                for row in raw
            ],
            "open": [float(row[1]) for row in raw],
            "high": [float(row[2]) for row in raw],
            "low": [float(row[3]) for row in raw],
            "close": [float(row[4]) for row in raw],
            "volume": [float(row[5]) for row in raw],
        }
    )
    return add_indicators(frame)


def btc_regime() -> int:
    """+1 when Bitcoin is above its 200 EMA, -1 when below. Read from 4h."""
    frame = candles_to_frame(fetch_klines("BTCUSDT", REGIME_INTERVAL, REGIME_DAYS))
    return 1 if frame["close"].iloc[-1] > frame["ema200"].iloc[-1] else -1


def detect(frame: pd.DataFrame) -> int:
    """Signal on the last closed candle: +1 long, -1 short, 0 nothing."""
    if len(frame) < 210:
        return 0

    fast, slow, trend = frame["ema20"], frame["ema50"], frame["ema200"]
    last = -1

    if pd.isna(trend.iloc[last]) or pd.isna(frame["atr"].iloc[last]):
        return 0

    long_ = (
        fast.iloc[last] > slow.iloc[last]
        and fast.iloc[last - 1] <= slow.iloc[last - 1]
        and frame["close"].iloc[last] > trend.iloc[last]
    )
    short = (
        fast.iloc[last] < slow.iloc[last]
        and fast.iloc[last - 1] >= slow.iloc[last - 1]
        and frame["close"].iloc[last] < trend.iloc[last]
    )

    return 1 if long_ else (-1 if short else 0)


def build_plan(symbol: str, frame: pd.DataFrame, side: int) -> dict:
    price = frame["close"].iloc[-1]
    atr = frame["atr"].iloc[-1]
    risk = SL_ATR * atr

    if side > 0:
        stop, target = price - risk, price + RR * risk
    else:
        stop, target = price + risk, price - RR * risk

    return {
        "symbol": symbol,
        "side": side,
        "price": price,
        "stop_pct": risk / price * 100,
        "target_pct": RR * risk / price * 100,
        "rsi": frame["rsi"].iloc[-1],
        "candle": frame["open_time"].iloc[-1],
    }


def format_message(plan: dict) -> str:
    arrow = "🟢 خرید (لانگ)" if plan["side"] > 0 else "🔴 فروش (شورت)"
    return (
        f"🔔 <b>سیگنال {plan['symbol']}</b>\n"
        "───────────────\n"
        f"<b>جهت:</b> {arrow}\n"
        f"<b>قیمت ورود:</b> {plan['price']:,.4f}\n"
        f"<b>حد ضرر:</b> −{plan['stop_pct']:.2f}٪\n"
        f"<b>حد سود:</b> +{plan['target_pct']:.2f}٪\n"
        "───────────────\n"
        f"کندل ۳۰ دقیقه: {plan['candle']} UTC · RSI {plan['rsi']:.0f}\n"
        "───────────────\n"
        f"{PAPER_BANNER}"
    )


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def main(argv: list[str]) -> int:
    do_send = "--send" in argv

    print("checking market regime...")
    regime = btc_regime()
    print(f"BTC regime: {'bullish (longs only)' if regime > 0 else 'bearish (shorts only)'}")

    symbols = top_symbols(limit=20)
    state = load_state()
    found = 0

    for index, symbol in enumerate(symbols, start=1):
        try:
            frame = candles_to_frame(fetch_klines(symbol, INTERVAL, LOOKBACK_DAYS))
            side = detect(frame)

            if side == 0 or side != regime:
                continue

            plan = build_plan(symbol, frame, side)
            if state.get(symbol) == plan["candle"]:
                print(f"[{index:2}/{len(symbols)}] {symbol:12} duplicate for {plan['candle']}, skipped")
                continue

            found += 1
            print(f"[{index:2}/{len(symbols)}] {symbol:12} SIGNAL {'LONG' if side > 0 else 'SHORT'}")

            if do_send:
                if send(format_message(plan)):
                    state[symbol] = plan["candle"]
                    save_state(state)
                    print(f"           sent to telegram")
            else:
                print(format_message(plan).replace("<b>", "").replace("</b>", ""))

        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the scan
            print(f"[{index:2}/{len(symbols)}] {symbol:12} failed: {exc}")

    print(f"\nscan finished - {found} new signal(s), {len(symbols)} symbols checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
