"""
Run every strategy in strategies.py across the 4h universe and rank them.

Fairness rules, identical for all 56 strategies:
  - entry at the OPEN of the bar after the signal (never the signal bar's close)
  - stop is checked before target inside the same bar (worst case)
  - fees 0.05%/side, slippage 0.02%/side, funding 0.01% per 8h held
  - first 70% of history in-sample, last 30% out-of-sample
  - three exit models, because a mean-reversion entry and a trend entry cannot
    be judged on the same target

A strategy only counts if it is positive in BOTH halves. Anything that works
only in-sample is an artefact and is rejected.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from backtest import FEE_RATE, FUNDING_PER_8H, SLIP_RATE
from features import add_all
from strategies import STRATEGIES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data4h")
REPORT_DIR = os.path.join(HERE, "reports")

BAR_HOURS = 4
IS_FRACTION = 0.70

EXIT_MODELS = {
    # tight target: right for mean reversion
    "fixed_2r": {"sl": 2.0, "rr": 2.0, "hold": 40, "trail": 0.0},
    # wider target: right for breakout / trend
    "fixed_3r": {"sl": 2.5, "rr": 3.0, "hold": 60, "trail": 0.0},
    # let the winner run: right for trend following
    "trail": {"sl": 2.5, "rr": 0.0, "hold": 60, "trail": 2.5},
}


def simulate(o, h, l, c, atr, sig, times, model):
    sl_atr = model["sl"]
    rr = model["rr"]
    max_hold = model["hold"]
    trail_atr = model["trail"]

    trades = []
    n = len(c)
    for i in np.flatnonzero(sig):
        side = int(sig[i])
        if i + 2 >= n:
            break
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        entry = o[i + 1]
        if entry <= 0:
            continue
        risk = sl_atr * a
        if risk <= 0:
            continue

        if side > 0:
            stop = entry - risk
            target = entry + rr * risk if rr else np.inf
        else:
            stop = entry + risk
            target = entry - rr * risk if rr else -np.inf

        best = entry
        armed = False
        last = min(i + 1 + max_hold, n - 1)
        exit_price, exit_idx = None, last

        for j in range(i + 1, last + 1):
            if side > 0:
                if trail_atr:
                    best = max(best, h[j])
                    if not armed and best - entry >= risk:
                        armed = True
                    if armed:
                        stop = max(stop, best - trail_atr * a)
                if l[j] <= stop:
                    exit_price, exit_idx = stop, j
                    break
                if h[j] >= target:
                    exit_price, exit_idx = target, j
                    break
            else:
                if trail_atr:
                    best = min(best, l[j])
                    if not armed and entry - best >= risk:
                        armed = True
                    if armed:
                        stop = min(stop, best + trail_atr * a)
                if h[j] >= stop:
                    exit_price, exit_idx = stop, j
                    break
                if l[j] <= target:
                    exit_price, exit_idx = target, j
                    break

        if exit_price is None:
            exit_price = c[last]

        held = exit_idx - i
        gross = (exit_price - entry) / entry * side
        cost = 2 * (FEE_RATE + SLIP_RATE) + FUNDING_PER_8H * (held * BAR_HOURS / 8)
        net = gross - cost
        risk_pct = risk / entry

        trades.append(
            {
                "t": times[i + 1],
                "net": net * 100,
                "R": net / risk_pct if risk_pct else 0.0,
            }
        )

    return trades


def stats(trades):
    if len(trades) < 20:
        return None
    frame = pd.DataFrame(trades)
    net = frame["net"] / 100
    wins, losses = net[net > 0], net[net <= 0]
    gl = abs(losses.sum())
    equity = (1 + net).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    return {
        "n": int(len(frame)),
        "win": round(float(len(wins) / len(frame) * 100), 1),
        "pf": round(float(wins.sum() / gl), 2) if gl else 0.0,
        "r": round(float(frame["R"].mean()), 4),
        "dd": round(float(dd * 100), 1),
    }


def main() -> int:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_4h.csv")))
    if not files:
        print("no data - run: python fetch4h.py 150 730")
        return 1

    print(f"loading {len(files)} symbols...")
    frames = {}
    for path in files:
        symbol = os.path.basename(path).split("_")[0]
        raw = pd.read_csv(path)
        if len(raw) < 400:
            continue
        frames[symbol] = add_all(raw)

    print(f"{len(frames)} symbols usable, ~{len(next(iter(frames.values())))} bars each\n")

    all_times = sorted({t for df in frames.values() for t in df["open_time"].tolist()})
    split = all_times[int(len(all_times) * IS_FRACTION)]
    print(f"in-sample ends at {split}\n")

    rows = []
    total = len(STRATEGIES) * len(EXIT_MODELS)
    done = 0

    for name, builder in STRATEGIES.items():
        for exit_name, model in EXIT_MODELS.items():
            trades = []
            for symbol, df in frames.items():
                try:
                    signal = builder(df).to_numpy(int)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name} failed on {symbol}: {exc}")
                    break
                trades.extend(
                    simulate(
                        df["open"].to_numpy(float),
                        df["high"].to_numpy(float),
                        df["low"].to_numpy(float),
                        df["close"].to_numpy(float),
                        df["atr"].to_numpy(float),
                        signal,
                        df["open_time"].tolist(),
                        model,
                    )
                )

            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total} combos done")

            if not trades:
                continue
            is_p = [t for t in trades if t["t"] < split]
            oos_p = [t for t in trades if t["t"] >= split]
            a, b, d = stats(trades), stats(is_p), stats(oos_p)
            if not (a and b and d):
                continue

            rows.append(
                {
                    "strategy": name,
                    "exit": exit_name,
                    "trades": a["n"],
                    "win": a["win"],
                    "pf": a["pf"],
                    "r": a["r"],
                    "dd": a["dd"],
                    "is_r": b["r"],
                    "is_pf": b["pf"],
                    "oos_r": d["r"],
                    "oos_pf": d["pf"],
                    "oos_n": d["n"],
                }
            )

    result = pd.DataFrame(rows)
    os.makedirs(REPORT_DIR, exist_ok=True)
    result.to_csv(os.path.join(REPORT_DIR, "strategy_sweep.csv"), index=False)

    result["both"] = (result["is_r"] > 0) & (result["oos_r"] > 0)
    survivors = result[result["both"]].sort_values("oos_r", ascending=False)

    print(f"\n{len(result)} combos measured · {len(survivors)} positive in both halves\n")
    header = (
        f"{'strategy':<30}{'exit':<10}{'n':>6}{'win%':>7}{'PF':>6}"
        f"{'avgR':>8}{'IS_R':>8}{'OOS_R':>8}{'OOS_PF':>8}"
    )
    print(header)
    print("-" * len(header))
    for _, row in survivors.head(25).iterrows():
        print(
            f"{row['strategy']:<30}{row['exit']:<10}{int(row['trades']):>6}"
            f"{row['win']:>7}{row['pf']:>6}{row['r']:>8}{row['is_r']:>8}"
            f"{row['oos_r']:>8}{row['oos_pf']:>8}"
        )

    if len(survivors) == 0:
        print("(none)")
        print("\nbest by out-of-sample expectancy regardless of in-sample:")
        for _, row in result.sort_values("oos_r", ascending=False).head(10).iterrows():
            print(
                f"  {row['strategy']:<30}{row['exit']:<10}"
                f"IS {row['is_r']:+.3f}R  OOS {row['oos_r']:+.3f}R  n={int(row['trades'])}"
            )

    with open(os.path.join(REPORT_DIR, "strategy_sweep.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "symbols": len(frames),
                "split": split,
                "combos": len(result),
                "survivors": len(survivors),
                "top": survivors.head(15).to_dict(orient="records"),
            },
            handle,
            indent=2,
            default=float,
        )

    print(f"\nreport: {os.path.join(REPORT_DIR, 'strategy_sweep.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
