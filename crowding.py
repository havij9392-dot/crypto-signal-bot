"""
Strategy 3 - "Crowding Hunt".

Thesis
------
Funding is the only free public number that shows how crowded one side of the book
is. When funding is far above its own recent normal, longs are paying heavily just
to stay in - the position is crowded and fragile, and it tends to unwind. When it is
far below, shorts are the crowded side. We take the other side of that crowd.

Why almost nobody trades this
-----------------------------
Retail screens show price and indicators. Funding lives in a separate API, updates
every 8 hours, and looks boring. Funds use it constantly; retail mostly does not.

Rules
-----
1. Sum funding into 8h buckets (some symbols switch between 4h and 8h periods, so
   bucketing keeps every symbol comparable).
2. Standardise each bucket against its own trailing 30 buckets (10 days).
3. If the score is above the threshold, go SHORT. Below minus threshold, go LONG.
4. Stop = 3 x ATR(1D), target = 2R, maximum hold 48 hours.
5. One position per symbol at a time.
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
FUNDING_DIR = os.path.join(DATA_DIR, "funding")
REPORT_DIR = os.path.join(HERE, "reports")

Z_LOOKBACK = 30        # 8h buckets = 10 days
STOP_ATR = 1.0         # one day's range: the crowd has 48h to unwind, not 2 weeks
RR = 2.0
MAX_HOLD_HOURS = 48
MAX_RISK_PCT = 12.0    # skip absurdly wide stops
IS_FRACTION = 0.70


def daily_atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
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


def load_frames(symbol: str):
    path = os.path.join(DATA_DIR, f"{symbol}_15m.csv")
    if not os.path.exists(path):
        return None, None

    raw = pd.read_csv(path)
    raw["dt"] = pd.to_datetime(raw["open_time"], utc=True)
    raw = raw.set_index("dt").sort_index()

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    hourly = raw.resample("1h").agg(agg).dropna()
    daily = raw.resample("1D").agg(agg).dropna()

    if len(hourly) < 200 or len(daily) < 20:
        return None, None

    daily["atr"] = daily_atr(daily)
    return hourly, daily


def funding_score(symbol: str) -> pd.Series | None:
    path = os.path.join(FUNDING_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None

    frame = pd.read_csv(path)
    frame["dt"] = pd.to_datetime(frame["funding_time"], utc=True)
    frame = frame.set_index("dt").sort_index()

    buckets = frame["funding_rate"].resample("8h").sum()
    buckets = buckets[buckets.index >= frame.index.min().floor("8h")]

    mean = buckets.rolling(Z_LOOKBACK).mean()
    std = buckets.rolling(Z_LOOKBACK).std()
    score = (buckets - mean) / std.replace(0, np.nan)
    return score.dropna()


def simulate(hourly: pd.DataFrame, daily: pd.DataFrame, score: pd.Series, threshold: float,
             momentum_filter: str = "none") -> list[dict]:
    """momentum_filter: 'none' | 'with' | 'against' (3-day price move vs trade side)."""
    h_open = hourly["open"].to_numpy(float)
    h_high = hourly["high"].to_numpy(float)
    h_low = hourly["low"].to_numpy(float)
    h_close = hourly["close"].to_numpy(float)
    h_times = hourly.index

    daily_closed = daily.copy()
    daily_closed["closed_at"] = daily_closed.index + pd.Timedelta(days=1)
    atr_table = daily_closed[["closed_at", "atr"]].reset_index(drop=True)

    trades: list[dict] = []
    last_exit = -1

    for stamp, value in score.items():
        if abs(value) < threshold:
            continue

        side = -1 if value > 0 else 1   # fade the crowded side

        entry_idx = int(h_times.searchsorted(stamp, side="left"))
        if entry_idx <= last_exit or entry_idx >= len(hourly) - 2:
            continue

        # daily ATR known at entry
        position = atr_table["closed_at"].searchsorted(stamp, side="right") - 1
        if position < 0:
            continue
        atr_value = atr_table["atr"].iloc[position]
        if not np.isfinite(atr_value) or atr_value <= 0:
            continue

        if momentum_filter != "none":
            lookback_idx = int(h_times.searchsorted(stamp - pd.Timedelta(days=3), side="left"))
            if lookback_idx >= entry_idx:
                continue
            move = (h_close[entry_idx - 1] - h_close[lookback_idx]) / h_close[lookback_idx]
            if momentum_filter == "against" and move * side > 0:
                continue
            if momentum_filter == "with" and move * side < 0:
                continue

        entry = h_open[entry_idx]
        risk = STOP_ATR * atr_value
        risk_pct = risk / entry * 100
        if risk_pct > MAX_RISK_PCT or risk_pct <= 0:
            continue

        stop = entry - risk if side > 0 else entry + risk
        target = entry + RR * risk if side > 0 else entry - RR * risk

        last = min(entry_idx + MAX_HOLD_HOURS, len(hourly) - 1)
        exit_price, exit_idx, reason = None, last, "time"

        for j in range(entry_idx, last + 1):
            if side > 0:
                if h_low[j] <= stop:
                    exit_price, exit_idx, reason = stop, j, "stop"
                    break
                if h_high[j] >= target:
                    exit_price, exit_idx, reason = target, j, "target"
                    break
            else:
                if h_high[j] >= stop:
                    exit_price, exit_idx, reason = stop, j, "stop"
                    break
                if h_low[j] <= target:
                    exit_price, exit_idx, reason = target, j, "target"
                    break

        if exit_price is None:
            exit_price = h_close[last]

        held = exit_idx - entry_idx
        gross = (exit_price - entry) / entry * side
        cost = 2 * (FEE_RATE + SLIP_RATE) + FUNDING_PER_8H * (held / 8)
        net = gross - cost

        trades.append(
            {
                "entry_time": h_times[entry_idx].isoformat(),
                "side": "long" if side > 0 else "short",
                "score": float(value),
                "net_pct": net * 100,
                "r_multiple": net / (risk / entry) if risk else 0.0,
                "bars": held,
                "reason": reason,
            }
        )
        last_exit = exit_idx

    return trades


def main(argv: list[str]) -> int:
    symbols = sorted(
        os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(DATA_DIR, "*_15m.csv"))
    )
    if len(argv) > 1 and argv[1] != "--all":
        symbols = [s for s in symbols if s in argv[1:]]

    print(f"symbols: {len(symbols)}")
    thresholds = [1.5, 2.0, 2.5]
    filters = ["none", "with", "against"]

    results = []
    for threshold in thresholds:
        for momentum in filters:
            all_trades: list[dict] = []
            for symbol in symbols:
                hourly, daily = load_frames(symbol)
                score = funding_score(symbol)
                if hourly is None or score is None:
                    continue
                all_trades.extend(simulate(hourly, daily, score, threshold, momentum))

            if not all_trades:
                continue

            ordered = sorted(t["entry_time"] for t in all_trades)
            split = ordered[int(len(ordered) * IS_FRACTION)]
            inside = [t for t in all_trades if t["entry_time"] < split]
            outside = [t for t in all_trades if t["entry_time"] >= split]

            results.append(
                {
                    "threshold": threshold,
                    "momentum": momentum,
                    "all": stats(all_trades),
                    "is": stats(inside),
                    "oos": stats(outside),
                    "split": split,
                    "trades_raw": all_trades,
                }
            )
            s, i, o = results[-1]["all"], results[-1]["is"], results[-1]["oos"]
            print(
                f"z>{threshold} momentum={momentum:<8} "
                f"n={s['trades']:5} IS PF={i['pf']:5.2f} OOS PF={o['pf']:5.2f} "
                f"win={s['win']:4.1f}% avgR={s['avg_r']:5.2f}"
            )

    if not results:
        print("no signals produced")
        return 0

    # choose on in-sample only, never on out-of-sample
    valid = [r for r in results if r["is"]["trades"] >= 100]
    if not valid:
        print("\nnot enough in-sample trades to choose a configuration")
        return 0

    best = max(valid, key=lambda r: r["is"]["exp"])
    print(f"\nchosen on in-sample: z>{best['threshold']} momentum={best['momentum']}")
    for label in ("is", "oos", "all"):
        s = best[label]
        print(
            f"  {label.upper():<4} n={s['trades']:5} win={s['win']:4.1f}% PF={s['pf']:5.2f} "
            f"exp={s['exp']:6.3f}% avgR={s['avg_r']:5.2f} maxDD={s['dd']:6.1f}%"
        )

    os.makedirs(REPORT_DIR, exist_ok=True)
    pd.DataFrame(best["trades_raw"]).to_csv(
        os.path.join(REPORT_DIR, "crowding_trades.csv"), index=False
    )
    with open(os.path.join(REPORT_DIR, "crowding_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "chosen": {"threshold": best["threshold"], "momentum": best["momentum"]},
                "is": best["is"],
                "oos": best["oos"],
                "all": best["all"],
                "grid": [
                    {
                        "threshold": r["threshold"],
                        "momentum": r["momentum"],
                        "is": r["is"],
                        "oos": r["oos"],
                    }
                    for r in results
                ],
            },
            handle,
            indent=2,
        )
    print(f"\nwritten to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
