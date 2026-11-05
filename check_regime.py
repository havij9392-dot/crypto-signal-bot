"""
Phase 2c - Robustness check.

One good out-of-sample number can be luck. This script asks a stricter question:
across ALL parameter combinations, does the market regime filter consistently
improve out-of-sample results, or did we just pick a lucky row?

A real effect shows up as a shift in the whole distribution, not one outlier.
"""

from __future__ import annotations

import glob
import itertools
import os

import pandas as pd

from backtest import DATA_DIR, IS_FRACTION, STRATEGIES, add_indicators
from research import prep, simulate_v2, stats


def main() -> int:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_4h.csv")))
    store, regime_by_time = prep(files)

    sample_times = store["BTCUSDT"]["time"]
    split_time = sample_times[int(len(sample_times) * IS_FRACTION)]

    signals = {
        (symbol, name): builder(
            add_indicators(pd.read_csv(os.path.join(DATA_DIR, f"{symbol}_4h.csv")))
        ).to_numpy(int)
        for symbol in store
        for name, builder in STRATEGIES.items()
    }

    grid = list(itertools.product([1.5, 2.0, 2.5], [1.5, 2.0, 3.0], [0.0, 2.0], [False, True]))

    rows = []
    for name in STRATEGIES:
        for sl_atr, rr, trail, use_regime in grid:
            is_trades, oos_trades = [], []
            for symbol, arrays in store.items():
                for trade in simulate_v2(
                    arrays,
                    signals[(symbol, name)],
                    regime_by_time,
                    sl_atr=sl_atr,
                    rr=rr,
                    trail_atr=trail,
                    use_regime=use_regime,
                ):
                    (is_trades if trade["entry_time"] < split_time else oos_trades).append(trade)

            is_stats, oos_stats = stats(is_trades), stats(oos_trades)
            if is_stats["trades"] < 100 or oos_stats["trades"] < 40:
                continue

            rows.append(
                {
                    "strategy": name,
                    "sl": sl_atr,
                    "rr": rr,
                    "trail": trail,
                    "regime": use_regime,
                    "is_pf": is_stats["pf"],
                    "is_exp": is_stats["exp"],
                    "oos_pf": oos_stats["pf"],
                    "oos_exp": oos_stats["exp"],
                    "oos_dd": oos_stats["dd"],
                    "oos_n": oos_stats["trades"],
                }
            )

    df = pd.DataFrame(rows)
    print(f"valid combos: {len(df)}\n")

    summary = (
        df.groupby(["strategy", "regime"])
        .agg(
            combos=("oos_pf", "size"),
            oos_pf_mean=("oos_pf", "mean"),
            oos_pf_median=("oos_pf", "median"),
            oos_exp_mean=("oos_exp", "mean"),
            pct_profitable=("oos_pf", lambda s: (s > 1).mean() * 100),
            worst_dd=("oos_dd", "min"),
        )
        .round(3)
    )
    print(summary.to_string())

    print("\n--- does the regime filter shift the whole distribution?")
    for name in STRATEGIES:
        subset = df[df["strategy"] == name]
        off = subset[~subset["regime"]]["oos_pf"]
        on = subset[subset["regime"]]["oos_pf"]
        if len(off) and len(on):
            print(
                f"{name:<14} regime off: median PF {off.median():.2f} "
                f"({(off > 1).mean() * 100:.0f}% profitable)   "
                f"regime on: median PF {on.median():.2f} "
                f"({(on > 1).mean() * 100:.0f}% profitable)"
            )

    out = os.path.join(DATA_DIR, "..", "reports", "robustness.csv")
    df.to_csv(out, index=False)
    print(f"\nwritten to {os.path.normpath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
