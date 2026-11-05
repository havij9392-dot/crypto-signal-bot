"""
Phase 6 - Live signal engine, 4h.

The strategy is the winner of the 56-strategy sweep (see sweep.py and
reports/strategy_sweep.csv):

    EMA20 crosses EMA50 on 4h, price on the correct side of EMA200,
    and Bitcoin itself must agree (close above/below its own 4h EMA200).

Measured on 138 symbols over 2 years, with real fees and slippage:
    3709 trades · 39.3% win · PF 1.27 · +0.143R per trade
    in-sample +0.114R · out-of-sample +0.191R  (out-of-sample is higher)
    92 of 138 symbols individually profitable
    still +0.100R after deleting the 10 best-contributing symbols

Exit: stop 2.5 x ATR, target 3R, give up after 60 bars (10 days).

The signal function is imported from strategies.py, so what runs live is
literally the same code the backtest measured. Runs are idempotent: one signal
per candle close, tracked in state/last_signals.json.

Usage:
    python scanner.py           # scan and print
    python scanner.py --send    # scan and push to Telegram
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from features import add_all
from notify import send
from okx import fetch_klines, top_symbols, volumes
from strategies import ema_cross_20_50

import account

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, "state")
STATE_FILE = os.path.join(STATE_DIR, "last_signals.json")

INTERVAL = "4h"
LOOKBACK_DAYS = 200      # 1200 bars: comfortably past the 200 EMA
WORKERS = 10

REGIME_INTERVAL = "4h"
REGIME_DAYS = 200

SL_ATR = 2.5
RR = 3.0
MAX_HOLD_BARS = 60       # 10 days on 4h

# Hard ceiling on leverage, chosen by Amir. RR does not affect liquidation -
# only the stop distance and the leverage do. With a median stop of 8.5%,
# 10x puts a losing trade at 85% of margin, which is heavy but never liquidated.
# Anything above 10x gets closed by the exchange before the stop is reached.
MAX_LEVERAGE = 10
LIQUIDATION_BUFFER = 0.80   # keep 20% of the liquidation distance unused

MIN_TURNOVER_USD = 3_000_000
MIN_STOP_PCT = 1.5       # below this the fee eats the edge (see reports/30m_verdict.md)

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
    return add_all(frame)


def btc_regime() -> int:
    """+1 when Bitcoin is above its own 4h EMA200, -1 when below."""
    frame = candles_to_frame(fetch_klines("BTCUSDT", REGIME_INTERVAL, REGIME_DAYS))
    return 1 if frame["close"].iloc[-1] > frame["ema200"].iloc[-1] else -1


def build_plan(symbol: str, frame: pd.DataFrame, side: int, acct: dict | None = None) -> dict | None:
    price = float(frame["close"].iloc[-1])
    atr = float(frame["atr"].iloc[-1])
    risk = SL_ATR * atr
    stop_pct = risk / price * 100

    if stop_pct < MIN_STOP_PCT:
        return None

    plan = {
        "symbol": symbol,
        "side": side,
        "price": price,
        "stop_pct": stop_pct,
        "target_pct": RR * risk / price * 100,
        "rsi": float(frame["rsi"].iloc[-1]),
        "candle": frame["open_time"].iloc[-1],
        "max_lev": max(1, min(MAX_LEVERAGE, int(LIQUIDATION_BUFFER * 100 / stop_pct))),
        "loss_at_max_lev": min(
            100.0,
            stop_pct * min(MAX_LEVERAGE, int(LIQUIDATION_BUFFER * 100 / stop_pct)),
        ),
        "sizing": None,
    }

    if acct:
        capital = float(acct.get("capital") or 0)
        if capital > 0:
            risk_pct, braked = account.effective_risk(acct)
            size = account.sizing(capital, risk_pct, stop_pct)
            if size:
                size["risk_pct"] = risk_pct * 100
                size["braked"] = braked
                size["capital"] = capital
                plan["sizing"] = size

    return plan


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
        thin = "⚠️ نقدشوندگی پایین — با حجم کم وارد شو\n"

    size = plan.get("sizing")
    if size:
        brake = " 🔻ترمز افت" if size.get("braked") else ""
        sizing_block = (
            "───────────────\n"
            f"💵 <b>سرمایه:</b> ${size['capital']:,.0f} · "
            f"ریسک {size['risk_pct']:.2f}٪{brake} = ${size['risk_usd']:,.0f}\n"
            f"<b>حجم پوزیشن:</b> ${size['notional']:,.0f}\n"
            f"<b>اهرم:</b> {size['leverage']}× · <b>مارجین:</b> ${size['margin']:,.0f}"
            f" ({size['margin_pct']:.1f}٪ حساب)\n"
            f"<b>اگر حد ضرر بخورد:</b> −${size['risk_usd']:,.0f}\n"
        )
    else:
        sizing_block = (
            "───────────────\n"
            f"اهرم: حداکثر {plan['max_lev']}× · "
            f"اگر حد ضرر بخورد {plan['loss_at_max_lev']:.0f}٪ مارجین\n"
            "برای دیدن حجم و مارجین، دستور /capital را بزن\n"
        )

    return (
        f"🔔 <b>سیگنال {plan['symbol']}</b>\n"
        "───────────────\n"
        f"<b>جهت:</b> {arrow}\n"
        f"<b>قیمت ورود:</b> {plan['price']:,.6f}\n"
        f"<b>حد ضرر:</b> −{plan['stop_pct']:.2f}٪\n"
        f"<b>حد سود:</b> +{plan['target_pct']:.2f}٪\n"
        "───────────────\n"
        f"{volume_line}"
        f"کندل ۴ ساعته: {plan['candle']} UTC · RSI {plan['rsi']:.0f}\n"
        f"{sizing_block}"
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


def scan_one(symbol: str, regime: int, acct: dict) -> dict | None:
    """Return a plan if this symbol qualifies, otherwise None."""
    frame = candles_to_frame(fetch_klines(symbol, INTERVAL, LOOKBACK_DAYS))
    if len(frame) < 220:
        return None

    side = int(ema_cross_20_50(frame).iloc[-1])
    if side == 0 or side != regime:
        return None
    return build_plan(symbol, frame, side, acct)


def main(argv: list[str]) -> int:
    do_send = "--send" in argv

    acct = account.load()
    if float(acct.get("capital") or 0) > 0:
        risk_pct, braked = account.effective_risk(acct)
        print(
            f"account: ${acct['capital']:,.0f} · risk {risk_pct * 100:.2f}%"
            f"{' (drawdown brake active)' if braked else ''}"
        )
    else:
        print("account: not set - signals will show percentages only (send /capital to the bot)")

    print("checking market regime...")
    regime = btc_regime()
    print(f"BTC regime: {'bullish (longs only)' if regime > 0 else 'bearish (shorts only)'}")

    symbols = top_symbols()

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
        futures = {pool.submit(scan_one, symbol, regime, acct): symbol for symbol in symbols}
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
