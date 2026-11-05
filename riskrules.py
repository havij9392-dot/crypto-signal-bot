"""
Does managing the risk level beat fixing it?

The 5-month run peaked at $3,537 and gave back 63% of it. That suggests the
problem is not the entries, it is that risk stayed at 1% of a growing account,
so the same percentage loss later removes far more dollars.

Six rules are tested on the identical 326 trades, in the identical order:

  fixed_1        risk 1% of current equity, never changes
  fixed_05       risk 0.5% of current equity
  fixed_dollar   risk 1% of the STARTING capital, so risk never compounds up
  step_half      risk halves every time equity is 50% above its checkpoint
  step_soft      risk drops 25% every time equity is 50% above its checkpoint
  drawdown_brake risk drops to a quarter while equity is 25% below its peak
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from backtest import FEE_RATE, FUNDING_PER_8H, SLIP_RATE
from features import add_all
from okx import volumes
from strategies import ema_cross_20_50

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data4h")
REPORT_DIR = os.path.join(HERE, "reports")

START = 1000.0
BASKET = 50
MONTHS = 5
SL_ATR, RR, HOLD = 2.5, 3.0, 60


def collect() -> pd.DataFrame:
    turn = volumes()
    ranked = sorted(
        (
            (os.path.basename(p).split("_")[0], turn.get(os.path.basename(p).split("_")[0][:-4], 0))
            for p in glob.glob(os.path.join(DATA_DIR, "*_4h.csv"))
        ),
        key=lambda item: -item[1],
    )
    btc = add_all(pd.read_csv(os.path.join(DATA_DIR, "BTCUSDT_4h.csv")))
    regime = dict(zip(btc["open_time"], np.where(btc["close"] > btc["ema200"], 1, -1)))

    rows = []
    for symbol, _ in ranked[:BASKET]:
        path = os.path.join(DATA_DIR, f"{symbol}_4h.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if len(df) < 400:
            continue
        df = add_all(df)

        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        atr = df["atr"].to_numpy(float)
        t = df["open_time"].tolist()

        signal = ema_cross_20_50(df).to_numpy(int)
        reg = np.array([regime.get(x, 0) for x in t])
        signal = np.where(signal == reg, signal, 0)

        i, n = 1, len(df)
        while i < n - 2:
            side = signal[i]
            if side == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                i += 1
                continue
            entry = o[i + 1]
            risk = SL_ATR * atr[i]
            if risk <= 0 or entry <= 0:
                i += 1
                continue

            stop, target = (
                (entry - risk, entry + RR * risk) if side > 0 else (entry + risk, entry - RR * risk)
            )
            last = min(i + 1 + HOLD, n - 1)
            exit_price, exit_idx = None, last

            for j in range(i + 1, last + 1):
                if side > 0:
                    if lo[j] <= stop:
                        exit_price, exit_idx = stop, j
                        break
                    if h[j] >= target:
                        exit_price, exit_idx = target, j
                        break
                else:
                    if h[j] >= stop:
                        exit_price, exit_idx = stop, j
                        break
                    if lo[j] <= target:
                        exit_price, exit_idx = target, j
                        break

            if exit_price is None:
                exit_price = c[last]

            held = exit_idx - i
            gross = (exit_price - entry) / entry * side
            cost = 2 * (FEE_RATE + SLIP_RATE) + FUNDING_PER_8H * (held * 4 / 8)
            net = gross - cost
            rows.append({"t": t[i + 1], "R": net / (risk / entry)})
            i = exit_idx + 1

    frame = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    return frame[frame["t"] > frame["t"].max()[:10] + ""] if False else frame


def recent(frame: pd.DataFrame) -> pd.DataFrame:
    cut = pd.to_datetime(frame["t"]).max() - pd.Timedelta(days=MONTHS * 30.5)
    return frame[pd.to_datetime(frame["t"]) > cut].reset_index(drop=True)


def run(rule: str, R: np.ndarray) -> dict:
    equity = START
    peak = START
    checkpoint = START
    risk = 0.01
    curve = [START]
    taken = 0

    for r in R:
        if rule == "fixed_1":
            risk = 0.01
        elif rule == "fixed_05":
            risk = 0.005
        elif rule == "fixed_dollar":
            risk = 0.01 * START / equity
        elif rule == "step_half":
            if equity >= checkpoint * 1.5:
                risk *= 0.5
                checkpoint = equity
        elif rule == "step_soft":
            if equity >= checkpoint * 1.5:
                risk *= 0.75
                checkpoint = equity
        elif rule == "drawdown_brake":
            base = 0.01
            risk = base * 0.25 if equity < peak * 0.75 else base

        equity *= 1 + r * risk
        taken += 1
        peak = max(peak, equity)
        curve.append(equity)

    curve = np.array(curve)
    running_peak = np.maximum.accumulate(curve)
    return {
        "rule": rule,
        "final": equity,
        "return_pct": (equity / START - 1) * 100,
        "peak": curve.max(),
        "max_dd": ((curve / running_peak) - 1).min() * 100,
        "worst_point": curve.min(),
        "trades": taken,
    }


def main() -> int:
    frame = collect()
    window = recent(frame)
    R = window["R"].to_numpy()
    print(f"{len(window)} trades · {window['t'].min()[:10]} to {window['t'].max()[:10]}")
    print(f"total {R.sum():+.1f}R · start ${START:,.0f}\n")

    rules = ["fixed_1", "fixed_05", "fixed_dollar", "step_half", "step_soft", "drawdown_brake"]
    results = [run(rule, R) for rule in rules]

    print(f"{'rule':<16}{'final':>10}{'return':>10}{'peak':>10}{'maxDD':>9}{'lowest':>10}")
    print("-" * 66)
    for row in results:
        print(
            f"{row['rule']:<16}{row['final']:>10,.0f}{row['return_pct']:>9.0f}%"
            f"{row['peak']:>10,.0f}{row['max_dd']:>8.0f}%{row['worst_point']:>10,.0f}"
        )

    best = max(results, key=lambda r: r["final"])
    print(f"\nbest final balance: {best['rule']} → ${best['final']:,.0f}")
    print(f"smallest drawdown : {min(results, key=lambda r: abs(r['max_dd']))['rule']}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(REPORT_DIR, "risk_rules.csv"), index=False)

    # equity curves for the chart
    out = {}
    for rule in ("fixed_1", "fixed_05", "step_half", "drawdown_brake"):
        equity, peak, checkpoint, risk = START, START, START, 0.01
        curve = [START]
        for r in R:
            if rule == "fixed_1":
                risk = 0.01
            elif rule == "fixed_05":
                risk = 0.005
            elif rule == "step_half":
                if equity >= checkpoint * 1.5:
                    risk *= 0.5
                    checkpoint = equity
            elif rule == "drawdown_brake":
                risk = 0.0025 if equity < peak * 0.75 else 0.01
            equity *= 1 + r * risk
            peak = max(peak, equity)
            curve.append(equity)
        out[rule] = [round(v) for v in curve]
    pd.DataFrame(out).to_csv(os.path.join(REPORT_DIR, "risk_curves.csv"), index=False)

    print(f"\nwritten to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
