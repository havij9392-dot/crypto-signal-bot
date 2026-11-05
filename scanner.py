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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from backtest import add_indicators
from notify import send
from okx import fetch_klines, top_symbols, volumes

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, "state")
STATE_FILE = os.path.join(STATE_DIR, "last_signals.json")

INTERVAL = "30m"            # entry timeframe
LOOKBACK_DAYS = 10          # 480 candles: enough for a 200 EMA, cheap to fetch
WORKERS = 8                 # parallel fetches, keeps a 200-symbol scan under 2 min

# The market regime is read from 4h, not 30m: a 200 EMA on 30m only covers four
# days and flips several times a week, which turns the filter into noise.
REGIME_INTERVAL = "4h"
REGIME_DAYS = 90

# Best of 48 stop/target/hold combinations measured on 30m by grid30.py.
# Nothing on 30m is profitable after fees; this is the least-bad one.
SL_ATR = 3.0
RR = 2.0

# A cross stays tradeable for this many candles (3 hours) as long as price is
# still on the right side of the 200 EMA. Waiting for the exact candle the cross
# happens misses most of them - the scan only wakes up twice an hour.
ENTRY_WINDOW = 4

# Coins below this 24h turnover are skipped: the spread alone eats the edge.
MIN_TURNOVER_USD = 3_000_000

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
    """Signal on the last closed candle: +1 long, -1 short, 0 nothing.

    A cross is valid for ENTRY_WINDOW candles as long as price is still on the
    correct side of the 200 EMA and no opposite cross happened in between.
    """
    if len(frame) < 210:
        return 0

    fast, slow, trend, close = (
        frame["ema20"],
        frame["ema50"],
        frame["ema200"],
        frame["close"],
    )
    last = -1

    if pd.isna(trend.iloc[last]) or pd.isna(frame["atr"].iloc[last]):
        return 0

    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    idx = pd.Series(range(len(frame)), index=frame.index)
    up_at = idx.where(cross_up).ffill()
    dn_at = idx.where(cross_dn).ffill()

    age_up = idx.iloc[last] - up_at.iloc[last]
    age_dn = idx.iloc[last] - dn_at.iloc[last]

    if pd.isna(age_up):
        age_up = 10**9
    if pd.isna(age_dn):
        age_dn = 10**9

    long_ = age_up < ENTRY_WINDOW and close.iloc[last] > trend.iloc[last] and age_up < age_dn
    short = age_dn < ENTRY_WINDOW and close.iloc[last] < trend.iloc[last] and age_dn < age_up

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

    volume = plan.get("volume_usd")
    if volume is None:
        volume_line = ""
    elif volume >= 1_000_000:
        volume_line = f"<b>حجم ۲۴ ساعته:</b> ${volume / 1_000_000:,.1f}M\n"
    else:
        volume_line = f"<b>حجم ۲۴ ساعته:</b> ${volume / 1_000:,.0f}K\n"

    thin = ""
    if volume is not None and volume < 5_000_000:
        thin = "⚠️ نقدشوندگی پایین — اسپرد پهن است، با حجم کم وارد شو\n"

    return (
        f"🔔 <b>سیگنال {plan['symbol']}</b>\n"
        "───────────────\n"
        f"<b>جهت:</b> {arrow}\n"
        f"<b>قیمت ورود:</b> {plan['price']:,.6f}\n"
        f"<b>حد ضرر:</b> −{plan['stop_pct']:.2f}٪\n"
        f"<b>حد سود:</b> +{plan['target_pct']:.2f}٪\n"
        "───────────────\n"
        f"{volume_line}"
        f"کندل ۳۰ دقیقه: {plan['candle']} UTC · RSI {plan['rsi']:.0f}\n"
        f"{thin}"
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


def scan_one(symbol: str, regime: int) -> dict | None:
    """Return a plan if this symbol qualifies, otherwise None."""
    frame = candles_to_frame(fetch_klines(symbol, INTERVAL, LOOKBACK_DAYS))
    side = detect(frame)
    if side == 0 or side != regime:
        return None
    return build_plan(symbol, frame, side)


def main(argv: list[str]) -> int:
    do_send = "--send" in argv

    print("checking market regime...")
    regime = btc_regime()
    print(f"BTC regime: {'bullish (longs only)' if regime > 0 else 'bearish (shorts only)'}")

    symbols = top_symbols()

    # One bulk call gives the 24h turnover for everything; skip the dead coins
    # before spending 200 candle requests on them.
    try:
        turnover = volumes()
    except Exception as exc:  # noqa: BLE001 - fall back to scanning everything
        print(f"turnover lookup failed ({exc}), scanning the full list")
        turnover = {}

    if turnover:
        liquid = [s for s in symbols if turnover.get(s[: -len("USDT")], 0) >= MIN_TURNOVER_USD]
        print(
            f"liquidity filter: {len(liquid)}/{len(symbols)} symbols "
            f"above ${MIN_TURNOVER_USD / 1_000_000:.0f}M daily turnover"
        )
        symbols = liquid

    print(f"scanning {len(symbols)} symbols on {INTERVAL}...")

    plans: list[dict] = []
    failures = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(scan_one, symbol, regime): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                plan = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the scan
                failures += 1
                print(f"  {symbol:<12} failed: {exc}")
                continue
            if plan:
                plans.append(plan)

    state = load_state()
    plans.sort(key=lambda item: item["symbol"])

    for plan in plans:
        plan["volume_usd"] = turnover.get(plan["symbol"][: -len("USDT")])

    found = 0

    for plan in plans:
        symbol = plan["symbol"]
        if state.get(symbol) == plan["candle"]:
            print(f"  {symbol:<12} duplicate for {plan['candle']}, skipped")
            continue

        found += 1
        print(f"  {symbol:<12} SIGNAL {'LONG' if plan['side'] > 0 else 'SHORT'}")

        if do_send:
            if send(format_message(plan)):
                state[symbol] = plan["candle"]
                save_state(state)
                print(f"  {symbol:<12} sent to telegram")
        else:
            print(format_message(plan).replace("<b>", "").replace("</b>", ""))

    print(
        f"\nscan finished - {found} new signal(s), "
        f"{len(symbols)} symbols checked, {failures} failed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
