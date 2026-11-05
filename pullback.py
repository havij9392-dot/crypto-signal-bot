"""
Strategy 2 - Trend following + pullback (Amir's spec).

1H trend  : EMA20 above EMA50, last two swing highs rising, last two swing lows rising
15m entry : price pulls back to EMA20/EMA50, a confirmation candle closes beyond the
            previous candle's extreme, RSI on the correct side of 50
Stop      : behind the last confirmed 15m swing point
Target    : 2R
Filter    : no trade when the 1H structure is not clearly up or down

Every value is computed from closed candles only. The 1H trend used at 15m bar t is
the trend of the last 1H candle that had already closed before t, so no future data
leaks into a signal.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from backtest import FEE_RATE, FUNDING_PER_8H, SLIP_RATE
from research import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
REPORT_DIR = os.path.join(HERE, "reports")

PIVOT_K = 3          # bars on each side that confirm a swing point
MAX_RISK_PCT = 3.0   # skip setups whose stop is further than 3% away
MIN_RISK_PCT = 0.3   # a stop closer than this sits inside the spread and fees eat it
MIN_RISK_ATR = 0.5   # ... and it must be at least half an ATR away
RR = 2.0
MAX_HOLD_15M = 96    # 24 hours
IS_FRACTION = 0.70


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False).mean()


def last_two_pivots(values: np.ndarray, k: int = PIVOT_K) -> tuple[np.ndarray, np.ndarray]:
    """For every bar, the most recent confirmed pivot and the one before it.

    A pivot at bar i is only usable at bar i + k, which is why it is stamped there.
    """
    n = len(values)
    confirmed_at: list[int] = []
    confirmed_val: list[float] = []

    for i in range(k, n - k):
        window = values[i - k : i + k + 1]
        if values[i] >= window.max():
            confirmed_at.append(i + k)
            confirmed_val.append(values[i])

    last = np.full(n, np.nan)
    previous = np.full(n, np.nan)
    if not confirmed_at:
        return last, previous

    at = np.asarray(confirmed_at)
    val = np.asarray(confirmed_val)
    count = np.searchsorted(at, np.arange(n), side="right")

    has_one = count >= 1
    last[has_one] = val[count[has_one] - 1]
    has_two = count >= 2
    previous[has_two] = val[count[has_two] - 2]
    return last, previous


def last_two_troughs(values: np.ndarray, k: int = PIVOT_K) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    confirmed_at: list[int] = []
    confirmed_val: list[float] = []

    for i in range(k, n - k):
        window = values[i - k : i + k + 1]
        if values[i] <= window.min():
            confirmed_at.append(i + k)
            confirmed_val.append(values[i])

    last = np.full(n, np.nan)
    previous = np.full(n, np.nan)
    if not confirmed_at:
        return last, previous

    at = np.asarray(confirmed_at)
    val = np.asarray(confirmed_val)
    count = np.searchsorted(at, np.arange(n), side="right")

    has_one = count >= 1
    last[has_one] = val[count[has_one] - 1]
    has_two = count >= 2
    previous[has_two] = val[count[has_two] - 2]
    return last, previous


def build(symbol: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """Return the 15m frame enriched with 1H trend and a series of entry signals."""
    path = os.path.join(DATA_DIR, f"{symbol}_15m.csv")
    if not os.path.exists(path):
        return None

    frame = pd.read_csv(path)
    frame["dt"] = pd.to_datetime(frame["open_time"], utc=True)
    frame = frame.set_index("dt").sort_index()

    hourly = frame.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    if len(hourly) < 220:
        return None

    hourly["ema20"] = ema(hourly["close"], 20)
    hourly["ema50"] = ema(hourly["close"], 50)

    high_last, high_prev = last_two_pivots(hourly["high"].to_numpy())
    low_last, low_prev = last_two_troughs(hourly["low"].to_numpy())

    up = (
        (hourly["ema20"] > hourly["ema50"])
        & (high_last > high_prev)
        & (low_last > low_prev)
    )
    down = (
        (hourly["ema20"] < hourly["ema50"])
        & (high_last < high_prev)
        & (low_last < low_prev)
    )
    hourly["trend"] = np.where(up, 1, np.where(down, -1, 0))

    # A 1H candle becomes usable only when it closes, so stamp it with its close time.
    trend_table = hourly[["trend"]].copy()
    trend_table["closed_at"] = trend_table.index + pd.Timedelta(hours=1)
    trend_table = trend_table.reset_index(drop=True)

    frame = frame.reset_index()
    frame = pd.merge_asof(
        frame.sort_values("dt"),
        trend_table.sort_values("closed_at"),
        left_on="dt",
        right_on="closed_at",
        direction="backward",
    )
    frame = frame.set_index("dt")

    frame["ema20"] = ema(frame["close"], 20)
    frame["ema50"] = ema(frame["close"], 50)
    frame["rsi"] = rsi(frame["close"])
    frame["atr"] = atr(frame)
    frame["prev_high"] = frame["high"].shift(1)
    frame["prev_low"] = frame["low"].shift(1)

    swing_low, _ = last_two_troughs(frame["low"].to_numpy())
    swing_high, _ = last_two_pivots(frame["high"].to_numpy())
    frame["swing_low"] = swing_low
    frame["swing_high"] = swing_high

    near_fast = (frame["low"] <= frame["ema20"] * 1.001) | (frame["low"] <= frame["ema50"] * 1.001)
    near_fast_short = (frame["high"] >= frame["ema20"] * 0.999) | (frame["high"] >= frame["ema50"] * 0.999)

    long_setup = (
        (frame["trend"] == 1)
        & near_fast
        & (frame["close"] > frame["ema20"])
        & (frame["close"] > frame["open"])
        & (frame["close"] > frame["prev_high"])
        & (frame["rsi"] > 50)
    )
    short_setup = (
        (frame["trend"] == -1)
        & near_fast_short
        & (frame["close"] < frame["ema20"])
        & (frame["close"] < frame["open"])
        & (frame["close"] < frame["prev_low"])
        & (frame["rsi"] < 50)
    )

    signal = np.where(long_setup, 1, np.where(short_setup, -1, 0))
    return frame, pd.Series(signal, index=frame.index)


def simulate(frame: pd.DataFrame, signal: pd.Series) -> list[dict]:
    o = frame["open"].to_numpy(float)
    h = frame["high"].to_numpy(float)
    l = frame["low"].to_numpy(float)
    c = frame["close"].to_numpy(float)
    atr_values = frame["atr"].to_numpy(float)
    swing_low = frame["swing_low"].to_numpy(float)
    swing_high = frame["swing_high"].to_numpy(float)
    times = [stamp.isoformat() for stamp in frame.index]
    sig = signal.to_numpy(int)

    trades: list[dict] = []
    i, n = 1, len(sig)

    while i < n - 2:
        side = sig[i]
        if side == 0:
            i += 1
            continue

        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue

        if side > 0:
            reference = swing_low[i]
            if not np.isfinite(reference):
                i += 1
                continue
            stop = reference - 0.1 * atr_values[i]
        else:
            reference = swing_high[i]
            if not np.isfinite(reference):
                i += 1
                continue
            stop = reference + 0.1 * atr_values[i]

        risk = abs(entry - stop)
        risk_pct_check = risk / entry * 100
        if (
            risk <= 0
            or risk_pct_check > MAX_RISK_PCT
            or risk_pct_check < MIN_RISK_PCT
            or risk < MIN_RISK_ATR * atr_values[i]
        ):
            i += 1
            continue

        target = entry + RR * risk if side > 0 else entry - RR * risk
        last = min(i + 1 + MAX_HOLD_15M, n - 1)
        exit_price, exit_idx, reason = None, last, "time"

        for j in range(i + 1, last + 1):
            if side > 0:
                if l[j] <= stop:
                    exit_price, exit_idx, reason = stop, j, "stop"
                    break
                if h[j] >= target:
                    exit_price, exit_idx, reason = target, j, "target"
                    break
            else:
                if h[j] >= stop:
                    exit_price, exit_idx, reason = stop, j, "stop"
                    break
                if l[j] <= target:
                    exit_price, exit_idx, reason = target, j, "target"
                    break

        if exit_price is None:
            exit_price = c[last]

        held = exit_idx - i
        gross = (exit_price - entry) / entry * side
        cost = 2 * (FEE_RATE + SLIP_RATE) + FUNDING_PER_8H * (held * 0.25 / 8)
        net = gross - cost
        risk_pct = risk / entry

        trades.append(
            {
                "entry_time": times[i + 1],
                "side": "long" if side > 0 else "short",
                "net_pct": net * 100,
                "r_multiple": net / risk_pct if risk_pct else 0.0,
                "bars": held,
                "reason": reason,
            }
        )
        i = exit_idx + 1

    return trades


def main(argv: list[str]) -> int:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_15m.csv")))
    if not files:
        print("no 15m data - run: python data.py 15m 365 --top 30")
        return 1

    symbols = [os.path.basename(f).split("_")[0] for f in files]
    if len(argv) > 1:
        symbols = [s for s in symbols if s in argv[1:]]

    all_trades: list[dict] = []
    per_symbol: list[dict] = []
    split_time = None

    for index, symbol in enumerate(symbols, start=1):
        built = build(symbol)
        if built is None:
            print(f"[{index:2}/{len(symbols)}] {symbol:12} skipped (not enough data)")
            continue
        frame, signal = built
        trades = simulate(frame, signal)
        all_trades.extend(trades)
        per_symbol.append({"symbol": symbol, "trades": len(trades), "stats": stats(trades)})
        print(f"[{index:2}/{len(symbols)}] {symbol:12} {len(trades):4} trades")

    if not all_trades:
        print("\nno trades produced - the filters are too strict or data is missing")
        return 0

    ordered = sorted(t["entry_time"] for t in all_trades)
    split_time = ordered[int(len(ordered) * IS_FRACTION)]

    inside = [t for t in all_trades if t["entry_time"] < split_time]
    outside = [t for t in all_trades if t["entry_time"] >= split_time]

    header = f"{'period':<10}{'trades':>8}{'win%':>7}{'PF':>7}{'exp%':>8}{'avgR':>7}{'maxDD%':>8}"
    print("\n" + header)
    print("-" * len(header))
    for label, subset in (("ALL", all_trades), ("IS", inside), ("OOS", outside)):
        s = stats(subset)
        if not s["trades"]:
            print(f"{label:<10}no trades")
            continue
        print(
            f"{label:<10}{s['trades']:>8}{s['win']:>7.1f}{s['pf']:>7.2f}"
            f"{s['exp']:>8.3f}{s['avg_r']:>7.2f}{s['dd']:>8.1f}"
        )

    long_count = sum(1 for t in all_trades if t["side"] == "long")
    print(f"\nlong: {long_count}   short: {len(all_trades) - long_count}")
    print(f"split at: {split_time}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    pd.DataFrame(all_trades).to_csv(os.path.join(REPORT_DIR, "pullback_trades.csv"), index=False)
    with open(os.path.join(REPORT_DIR, "pullback_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "symbols": len(symbols),
                "total_trades": len(all_trades),
                "split_time": split_time,
                "all": stats(all_trades),
                "is": stats(inside),
                "oos": stats(outside),
                "per_symbol": per_symbol,
            },
            handle,
            indent=2,
        )
    print(f"\nwritten to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
