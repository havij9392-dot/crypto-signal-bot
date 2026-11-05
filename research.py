"""
Phase 2b - Research pass.

The first backtest showed all three strategies making money in-sample and losing
money out-of-sample. That is the signature of curve-fitting, so before adding any
new idea we test two structural fixes:

  1. Market regime filter - only take longs when BTC is above its 200 EMA and
     only take shorts when it is below. Crypto moves as one correlated block, so
     fighting BTC is the fastest way to lose.
  2. Trailing stop - a fixed 2R target caps the rare huge winners that pay for
     all the small losses.

Parameters are chosen on in-sample data ONLY. Out-of-sample is reported once, as
a verdict, and is never used to pick anything.
"""

from __future__ import annotations

import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

from backtest import (
    DATA_DIR,
    FEE_RATE,
    FUNDING_PER_8H,
    IS_FRACTION,
    MAX_HOLD,
    REPORT_DIR,
    SLIP_RATE,
    STRATEGIES,
    add_indicators,
)

MIN_TRADES = 120


def prep(files: list[str]) -> tuple[dict, dict]:
    """Load every symbol once and precompute indicators + the BTC regime series."""
    store: dict[str, dict] = {}
    for path in files:
        symbol = os.path.basename(path).split("_")[0]
        df = add_indicators(pd.read_csv(path))
        store[symbol] = {
            "o": df["open"].to_numpy(float),
            "h": df["high"].to_numpy(float),
            "l": df["low"].to_numpy(float),
            "c": df["close"].to_numpy(float),
            "atr": df["atr"].to_numpy(float),
            "time": df["open_time"].tolist(),
        }

    btc = add_indicators(pd.read_csv(os.path.join(DATA_DIR, "BTCUSDT_4h.csv")))
    regime = np.where(btc["close"] > btc["ema200"], 1, -1)
    regime_by_time = dict(zip(btc["open_time"].tolist(), regime.tolist()))

    return store, regime_by_time


def simulate_v2(
    arrays: dict,
    signal: np.ndarray,
    regime_by_time: dict,
    *,
    sl_atr: float,
    rr: float,
    trail_atr: float,
    use_regime: bool,
) -> list[dict]:
    o, h, l, c = arrays["o"], arrays["h"], arrays["l"], arrays["c"]
    atr, times = arrays["atr"], arrays["time"]
    sig = signal.copy()

    if use_regime:
        for index in range(len(sig)):
            if sig[index] == 0:
                continue
            direction = regime_by_time.get(times[index])
            if direction is None or direction != sig[index]:
                sig[index] = 0

    trades: list[dict] = []
    i, n = 1, len(sig)

    while i < n - 2:
        side = int(sig[i])
        if side == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue

        entry = o[i + 1]
        risk = sl_atr * atr[i]
        if risk <= 0 or entry <= 0:
            i += 1
            continue

        if side > 0:
            stop, target = entry - risk, entry + rr * risk
        else:
            stop, target = entry + risk, entry - rr * risk

        last = min(i + 1 + MAX_HOLD, n - 1)
        exit_price, exit_idx, reason = None, last, "time"
        best = entry

        for j in range(i + 1, last + 1):
            if side > 0:
                if l[j] <= stop:
                    exit_price, exit_idx, reason = stop, j, "stop"
                    break
                if h[j] >= target:
                    exit_price, exit_idx, reason = target, j, "target"
                    break
                if trail_atr > 0:
                    best = max(best, h[j])
                    stop = max(stop, best - trail_atr * atr[j])
            else:
                if h[j] >= stop:
                    exit_price, exit_idx, reason = stop, j, "stop"
                    break
                if l[j] <= target:
                    exit_price, exit_idx, reason = target, j, "target"
                    break
                if trail_atr > 0:
                    best = min(best, l[j])
                    stop = min(stop, best + trail_atr * atr[j])

        if exit_price is None:
            exit_price = c[last]

        held = exit_idx - i
        gross = (exit_price - entry) / entry * side
        cost = 2 * (FEE_RATE + SLIP_RATE) + FUNDING_PER_8H * (held * 4 / 8)
        net = gross - cost
        risk_pct = risk / entry

        trades.append(
            {
                "entry_time": times[i + 1],
                "net_pct": net * 100,
                "r_multiple": net / risk_pct if risk_pct else 0.0,
                "bars": held,
                "reason": reason,
                "side": "long" if side > 0 else "short",
            }
        )
        i = exit_idx + 1

    return trades


def stats(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "pf": 0.0, "exp": 0.0, "win": 0.0, "avg_r": 0.0, "dd": 0.0}

    net = np.array([t["net_pct"] for t in trades]) / 100
    wins, losses = net[net > 0], net[net <= 0]
    equity = (1 + net).cumprod()
    peak = np.maximum.accumulate(equity)

    return {
        "trades": len(net),
        "pf": float(wins.sum() / abs(losses.sum())) if losses.sum() else 99.0,
        "exp": float(net.mean() * 100),
        "win": float(len(wins) / len(net) * 100),
        "avg_r": float(np.mean([t["r_multiple"] for t in trades])),
        "dd": float((equity / peak - 1).min() * 100),
    }


def main() -> int:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_4h.csv")))
    if not files:
        print("no data - run: python data.py 4h 730")
        return 1

    store, regime_by_time = prep(files)
    sample_times = store["BTCUSDT"]["time"]
    split_time = sample_times[int(len(sample_times) * IS_FRACTION)]

    # Build every signal series once - the grid only changes exit logic.
    signals = {
        (symbol, name): builder(add_indicators(pd.read_csv(
            os.path.join(DATA_DIR, f"{symbol}_4h.csv")))).to_numpy(int)
        for symbol in store
        for name, builder in STRATEGIES.items()
    }

    grid = list(
        itertools.product([1.5, 2.0, 2.5], [1.5, 2.0, 3.0], [0.0, 2.0], [False, True])
    )
    print(f"grid: {len(grid)} combos x {len(STRATEGIES)} strategies x {len(store)} symbols\n")

    best_rows = []
    for name in STRATEGIES:
        candidates = []
        for sl_atr, rr, trail, use_regime in grid:
            is_trades: list[dict] = []
            oos_trades: list[dict] = []

            for symbol, arrays in store.items():
                result = simulate_v2(
                    arrays,
                    signals[(symbol, name)],
                    regime_by_time,
                    sl_atr=sl_atr,
                    rr=rr,
                    trail_atr=trail,
                    use_regime=use_regime,
                )
                for trade in result:
                    (is_trades if trade["entry_time"] < split_time else oos_trades).append(trade)

            is_stats = stats(is_trades)
            if is_stats["trades"] < MIN_TRADES:
                continue
            candidates.append(
                {
                    "strategy": name,
                    "sl_atr": sl_atr,
                    "rr": rr,
                    "trail": trail,
                    "regime": use_regime,
                    "is": is_stats,
                    "oos": stats(oos_trades),
                }
            )

        candidates.sort(key=lambda row: row["is"]["exp"], reverse=True)
        if candidates:
            best_rows.append(candidates[0])
            print(f"--- {name}: {len(candidates)} valid combos, best in-sample chosen")

    header = (
        f"{'strategy':<14}{'sl':>5}{'rr':>5}{'trail':>7}{'reg':>5}"
        f"{'IS n':>7}{'IS PF':>7}{'IS exp%':>9}"
        f"{'OOS n':>7}{'OOS PF':>8}{'OOS exp%':>10}{'OOS win%':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in best_rows:
        print(
            f"{row['strategy']:<14}{row['sl_atr']:>5}{row['rr']:>5}"
            f"{row['trail']:>7}{'on' if row['regime'] else 'off':>5}"
            f"{row['is']['trades']:>7}{row['is']['pf']:>7.2f}{row['is']['exp']:>9.3f}"
            f"{row['oos']['trades']:>7}{row['oos']['pf']:>8.2f}"
            f"{row['oos']['exp']:>10.3f}{row['oos']['win']:>10.1f}"
        )

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "research.json"), "w", encoding="utf-8") as handle:
        json.dump({"split_time": split_time, "best": best_rows}, handle, indent=2)

    print(f"\nwritten to {os.path.join(REPORT_DIR, 'research.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
