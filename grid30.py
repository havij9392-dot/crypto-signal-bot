"""
Phase 5b - Is the 30m timeframe salvageable at all?

entry_window.py showed the current 30m setup loses money (PF 0.93 over 12
months, 3688 trades). The exit breakdown explains why: 14% of trades hit the
48h time limit at only +0.74R, so a target that is 7.5 ATR away almost never
gets reached in two days.

This sweeps stop distance x reward ratio x max hold and reports in-sample and
out-of-sample separately. If nothing is positive in both, 30m is the problem,
not the parameters.
"""

from __future__ import annotations

import glob
import itertools
import os

import numpy as np
import pandas as pd

from backtest import FEE_RATE, FUNDING_PER_8H, SLIP_RATE
from entry_window import (
    BAR_MINUTES,
    DATA_DIR,
    IS_FRACTION,
    btc_regime_series,
    load_30m,
    window_signal,
)

WINDOW = 4
SL_GRID = [1.5, 2.0, 2.5, 3.0]
RR_GRID = [1.5, 2.0, 2.5, 3.0]
HOLD_GRID = [48, 96, 240]      # 1d, 2d, 5d on 30m bars


def run(df, signal, regime, sl_atr, rr, max_hold):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    times = df["open_time"].tolist()
    sig = signal.to_numpy(int)
    reg = regime.to_numpy(int)

    out = []
    i, n = 1, len(df)
    while i < n - 2:
        side = sig[i]
        if side == 0 or side != reg[i] or not np.isfinite(atr[i]) or atr[i] <= 0:
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

        last = min(i + 1 + max_hold, n - 1)
        exit_price, exit_idx, reason = None, last, "time"

        for j in range(i + 1, last + 1):
            if side > 0:
                if lo[j] <= stop:
                    exit_price, exit_idx, reason = stop, j, "stop"
                    break
                if h[j] >= target:
                    exit_price, exit_idx, reason = target, j, "target"
                    break
            else:
                if h[j] >= stop:
                    exit_price, exit_idx, reason = stop, j, "stop"
                    break
                if lo[j] <= target:
                    exit_price, exit_idx, reason = target, j, "target"
                    break

        if exit_price is None:
            exit_price = c[last]

        held = exit_idx - i
        gross = (exit_price - entry) / entry * side
        cost = 2 * (FEE_RATE + SLIP_RATE) + FUNDING_PER_8H * (held * BAR_MINUTES / 480)
        net = gross - cost
        risk_pct = risk / entry

        out.append(
            {
                "entry_time": times[i + 1],
                "net_pct": net * 100,
                "r_multiple": net / risk_pct if risk_pct else 0.0,
            }
        )
        i = exit_idx + 1

    return out


def stats(trades):
    if not trades:
        return {"n": 0, "pf": 0.0, "r": 0.0, "win": 0.0}
    frame = pd.DataFrame(trades)
    net = frame["net_pct"] / 100
    wins, losses = net[net > 0], net[net <= 0]
    gl = abs(losses.sum())
    return {
        "n": len(frame),
        "pf": round(wins.sum() / gl, 2) if gl else 0.0,
        "r": round(frame["r_multiple"].mean(), 3),
        "win": round(len(wins) / len(frame) * 100, 1),
    }


def main() -> int:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_15m.csv")))
    frames = {}
    for path in files:
        symbol = os.path.basename(path).split("_")[0]
        df = load_30m(path)
        frames[symbol] = (df, btc_regime_series(df["ts"]), window_signal(df, WINDOW))

    all_times = sorted({t for df, _, _ in frames.values() for t in df["open_time"].tolist()})
    split = all_times[int(len(all_times) * IS_FRACTION)]

    print(f"symbols {len(frames)} · window {WINDOW} · split {split}\n")
    header = (
        f"{'SL':>5}{'RR':>5}{'hold':>6}{'trades':>8}{'win%':>7}{'PF':>7}{'avgR':>8}"
        f"   {'IS_PF':>7}{'IS_R':>7}{'OOS_PF':>8}{'OOS_R':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for sl_atr, rr, hold in itertools.product(SL_GRID, RR_GRID, HOLD_GRID):
        trades = []
        for df, regime, signal in frames.values():
            trades.extend(run(df, signal, regime, sl_atr, rr, hold))

        is_p = [t for t in trades if t["entry_time"] < split]
        oos_p = [t for t in trades if t["entry_time"] >= split]
        a, b, d = stats(trades), stats(is_p), stats(oos_p)

        rows.append((sl_atr, rr, hold, a, b, d))
        flag = "  <-- both sides positive" if b["r"] > 0 and d["r"] > 0 else ""
        print(
            f"{sl_atr:>5}{rr:>5}{hold:>6}{a['n']:>8}{a['win']:>7}{a['pf']:>7}{a['r']:>8}"
            f"   {b['pf']:>7}{b['r']:>7}{d['pf']:>8}{d['r']:>8}{flag}"
        )

    survivors = [r for r in rows if r[4]["r"] > 0 and r[5]["r"] > 0]
    print(f"\nconfigs positive in BOTH halves: {len(survivors)} / {len(rows)}")
    for sl_atr, rr, hold, a, b, d in sorted(survivors, key=lambda r: -(r[4]["r"] + r[5]["r"]))[:5]:
        print(
            f"  SL {sl_atr} ATR · RR {rr} · hold {hold} bars · "
            f"IS {b['r']:+.3f}R · OOS {d['r']:+.3f}R · {a['n']} trades"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
