"""
Robustness checks for the strategies that survived the sweep.

Being positive in one 70/30 split is weak evidence. A real edge has to survive:
  - several different split points
  - a year-by-year breakdown
  - removing the handful of symbols that contributed the most
  - a per-symbol view, so one lucky coin cannot carry the whole result

Anything that fails these is reported as a coincidence, not an edge.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from features import add_all
from strategies import STRATEGIES
from sweep import EXIT_MODELS, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data4h")
REPORT_DIR = os.path.join(HERE, "reports")

FINALISTS = [
    ("rsi14_pullback", "fixed_3r"),
    ("bollinger_bounce", "fixed_3r"),
    ("ema_cross_20_50", "fixed_3r"),
    ("ichimoku_tk", "fixed_3r"),
    ("rsi14_pullback", "fixed_2r"),
    ("guppy_mma", "fixed_3r"),
]


def load_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_4h.csv"))):
        symbol = os.path.basename(path).split("_")[0]
        raw = pd.read_csv(path)
        if len(raw) < 400:
            continue
        frames[symbol] = add_all(raw)
    return frames


def collect(frames, name, exit_name):
    builder = STRATEGIES[name]
    model = EXIT_MODELS[exit_name]
    trades = []
    for symbol, df in frames.items():
        signal = builder(df).to_numpy(int)
        for trade in simulate(
            df["open"].to_numpy(float),
            df["high"].to_numpy(float),
            df["low"].to_numpy(float),
            df["close"].to_numpy(float),
            df["atr"].to_numpy(float),
            signal,
            df["open_time"].tolist(),
            model,
        ):
            trade["symbol"] = symbol
            trades.append(trade)
    return pd.DataFrame(trades)


def expectancy(frame: pd.DataFrame) -> tuple[float, float, int]:
    if len(frame) < 20:
        return 0.0, 0.0, len(frame)
    net = frame["net"] / 100
    wins, losses = net[net > 0], net[net <= 0]
    gl = abs(losses.sum())
    pf = float(wins.sum() / gl) if gl else 0.0
    return round(float(frame["R"].mean()), 4), round(pf, 2), len(frame)


def main() -> int:
    frames = load_frames()
    print(f"{len(frames)} symbols\n")

    times = sorted({t for df in frames.values() for t in df["open_time"].tolist()})

    print("=" * 78)
    print("1. SPLIT-POINT STABILITY  (avgR of the out-of-sample part)")
    print("=" * 78)
    header = f"{'strategy':<22}{'exit':<10}" + "".join(f"{p:>10}" for p in ("50/50", "60/40", "70/30", "80/20"))
    print(header)
    print("-" * len(header))

    for name, exit_name in FINALISTS:
        frame = collect(frames, name, exit_name)
        row = f"{name:<22}{exit_name:<10}"
        for fraction in (0.50, 0.60, 0.70, 0.80):
            cut = times[int(len(times) * fraction)]
            oos = frame[frame["t"] >= cut]
            r, _, _ = expectancy(oos)
            row += f"{r:>10}"
        print(row)

    print()
    print("=" * 78)
    print("2. YEAR BY YEAR  (avgR per calendar year)")
    print("=" * 78)
    header = f"{'strategy':<22}{'exit':<10}" + "".join(f"{y:>10}" for y in ("2024", "2025", "2026"))
    print(header)
    print("-" * len(header))
    for name, exit_name in FINALISTS:
        frame = collect(frames, name, exit_name)
        frame["year"] = frame["t"].str[:4]
        row = f"{name:<22}{exit_name:<10}"
        for year in ("2024", "2025", "2026"):
            r, _, _ = expectancy(frame[frame["year"] == year])
            row += f"{r:>10}"
        print(row)

    print()
    print("=" * 78)
    print("3. CONCENTRATION  (drop the 5 best-contributing symbols)")
    print("=" * 78)
    header = f"{'strategy':<22}{'exit':<10}{'all':>9}{'drop5':>9}{'drop10':>9}{'symbols+':>10}"
    print(header)
    print("-" * len(header))
    for name, exit_name in FINALISTS:
        frame = collect(frames, name, exit_name)
        total, _, _ = expectancy(frame)

        contribution = frame.groupby("symbol")["R"].sum().sort_values(ascending=False)
        keep5 = frame[~frame["symbol"].isin(contribution.head(5).index)]
        keep10 = frame[~frame["symbol"].isin(contribution.head(10).index)]
        d5, _, _ = expectancy(keep5)
        d10, _, _ = expectancy(keep10)

        per_symbol = frame.groupby("symbol")["R"].mean()
        positive = int((per_symbol > 0).sum())
        print(
            f"{name:<22}{exit_name:<10}{total:>9}{d5:>9}{d10:>9}"
            f"{positive:>6}/{len(per_symbol):<3}"
        )

    print()
    print("=" * 78)
    print("4. PER-SYMBOL SPREAD  (how many symbols are individually profitable)")
    print("=" * 78)
    for name, exit_name in FINALISTS:
        frame = collect(frames, name, exit_name)
        per_symbol = frame.groupby("symbol")["R"].mean()
        print(
            f"{name:<22}{exit_name:<10} "
            f"median {per_symbol.median():+.3f}R  "
            f"p25 {per_symbol.quantile(0.25):+.3f}  "
            f"p75 {per_symbol.quantile(0.75):+.3f}  "
            f"worst {per_symbol.min():+.3f}  best {per_symbol.max():+.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
