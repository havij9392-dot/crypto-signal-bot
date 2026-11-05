"""
Experiment - trend following with a crowding filter.

The 4h EMA-cross + BTC regime strategy is the only thing so far that made money in
BOTH halves of the sample. Its weakness is late entries: by the time the cross
happens, the move is often crowded.

Funding tells us exactly how crowded. So: keep the trend signal, but refuse to buy
when longs are already paying a premium, and refuse to short when shorts are.

This is the professional habit retail skips - not a new indicator, a veto.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from backtest import FEE_RATE, FUNDING_PER_8H, SLIP_RATE, add_indicators, strat_ema_cross
from crowding import funding_score
from research import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SL_ATR = 2.5
RR = 3.0
MAX_HOLD = 30
IS_FRACTION = 0.70


def simulate(frame: pd.DataFrame, signal: np.ndarray, z_series: np.ndarray,
             z_threshold: float | None) -> list[dict]:
    o = frame["open"].to_numpy(float)
    h = frame["high"].to_numpy(float)
    l = frame["low"].to_numpy(float)
    c = frame["close"].to_numpy(float)
    atr = frame["atr"].to_numpy(float)
    times = frame["open_time"].tolist()

    trades: list[dict] = []
    i, n = 1, len(signal)

    while i < n - 2:
        side = int(signal[i])
        if side == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue

        if z_threshold is not None and np.isfinite(z_series[i]):
            if side > 0 and z_series[i] > z_threshold:
                i += 1
                continue
            if side < 0 and z_series[i] < -z_threshold:
                i += 1
                continue

        entry = o[i + 1]
        risk = SL_ATR * atr[i]
        if risk <= 0 or entry <= 0:
            i += 1
            continue

        stop = entry - risk if side > 0 else entry + risk
        target = entry + RR * risk if side > 0 else entry - RR * risk

        last = min(i + 1 + MAX_HOLD, n - 1)
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
        cost = 2 * (FEE_RATE + SLIP_RATE) + FUNDING_PER_8H * (held * 4 / 8)
        net = gross - cost

        trades.append(
            {
                "entry_time": times[i + 1],
                "net_pct": net * 100,
                "r_multiple": net / (risk / entry) if risk else 0.0,
                "bars": held,
                "reason": reason,
                "side": "long" if side > 0 else "short",
            }
        )
        i = exit_idx + 1

    return trades


def main() -> int:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_4h.csv")))
    btc = add_indicators(pd.read_csv(os.path.join(DATA_DIR, "BTCUSDT_4h.csv")))
    regime_by_time = dict(
        zip(btc["open_time"].tolist(), np.where(btc["close"] > btc["ema200"], 1, -1).tolist())
    )

    prepared = []
    for path in files:
        symbol = os.path.basename(path).split("_")[0]
        frame = add_indicators(pd.read_csv(path))
        signal = strat_ema_cross(frame).to_numpy(int).copy()

        # apply the BTC regime filter
        for index in range(len(signal)):
            if signal[index] and regime_by_time.get(frame["open_time"].iloc[index]) != signal[index]:
                signal[index] = 0

        score = funding_score(symbol)
        if score is None:
            continue

        bucket_time = pd.to_datetime(frame["open_time"], utc=True)
        positions = score.index.searchsorted(bucket_time, side="right") - 1
        z_series = np.where(
            positions >= 0,
            score.to_numpy()[np.clip(positions, 0, len(score) - 1)],
            np.nan,
        )
        prepared.append((frame, signal, z_series))

    print(f"symbols: {len(prepared)}\n")
    print(f"{'funding filter':<20}{'trades':>8}{'win%':>7}{'IS PF':>8}{'OOS PF':>8}{'avgR':>7}{'maxDD%':>8}")
    print("-" * 68)

    rows = []
    for threshold in [None, 0.5, 1.0, 1.5, 2.0]:
        trades = []
        for frame, signal, z_series in prepared:
            trades.extend(simulate(frame, signal, z_series, threshold))

        ordered = sorted(t["entry_time"] for t in trades)
        split = ordered[int(len(ordered) * IS_FRACTION)]
        inside = [t for t in trades if t["entry_time"] < split]
        outside = [t for t in trades if t["entry_time"] >= split]

        a, i, o = stats(trades), stats(inside), stats(outside)
        label = "none" if threshold is None else f"skip if |z|>{threshold}"
        print(f"{label:<20}{a['trades']:>8}{a['win']:>7.1f}{i['pf']:>8.2f}{o['pf']:>8.2f}"
              f"{a['avg_r']:>7.2f}{a['dd']:>8.1f}")
        rows.append({"threshold": threshold, "all": a, "is": i, "oos": o})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
