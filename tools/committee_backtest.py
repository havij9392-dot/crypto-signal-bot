"""
Does each committee agent actually earn its seat?

An agent that sounds clever but does not improve the measured edge is a
liability: it removes trades, and every removed trade is a trade the base
strategy would have taken. So each one is measured alone, on the same trades,
with the same costs, and split in-sample / out-of-sample.

Method, identical for every agent:
  - the base signal is the live one (EMA20/50 cross + EMA200 + BTC regime)
  - entry at the OPEN of the bar after the signal
  - stop 2.5 x ATR, target 3R, give up after 60 bars, stop checked first
  - fees 0.05%/side, slippage 0.02%/side, funding 0.01% per 8h
  - votes are read at the signal bar, so nothing is known that was not known
  - first 70% of the calendar is in-sample, the last 30% out-of-sample

An agent is kept only if it improves expectancy in BOTH halves. Anything that
helps in-sample and hurts out-of-sample is an artefact.

Usage:
    python tools/committee_backtest.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import committee as C
from backtest import FEE_RATE, FUNDING_PER_8H, SLIP_RATE
from features import add_all
from strategies import ema_cross_20_50

DATA_DIR = os.path.join(HERE, "data4h")
REPORT_DIR = os.path.join(HERE, "reports")

SL_ATR, RR, MAX_HOLD, BAR_HOURS = 2.5, 3.0, 60, 4
IS_FRACTION = 0.70


def load(path: str) -> pd.DataFrame | None:
    try:
        raw = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return None
    if len(raw) < 400:
        return None
    return add_all(raw.rename(columns={"open_time": "time"}))


def one_trade(o, h, l, c, atr, times, i: int, side: int):
    """Simulate the single trade the base strategy would take at bar i."""
    n = len(c)
    if i + 2 >= n:
        return None
    a = atr[i]
    if not np.isfinite(a) or a <= 0:
        return None
    entry = o[i + 1]
    if entry <= 0:
        return None

    risk = SL_ATR * a
    if side > 0:
        stop, target = entry - risk, entry + RR * risk
    else:
        stop, target = entry + risk, entry - RR * risk

    last = min(i + 1 + MAX_HOLD, n - 1)
    exit_price, exit_idx = None, last

    for j in range(i + 1, last + 1):
        if side > 0:
            if l[j] <= stop:
                exit_price, exit_idx = stop, j
                break
            if h[j] >= target:
                exit_price, exit_idx = target, j
                break
        else:
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

    return {"t": times[i + 1], "i": i, "side": side,
            "r": net / risk_pct if risk_pct else 0.0}


def breadth_map(paths: list[str]) -> dict:
    """Share of the universe above its own EMA200, at every bar.

    Computed straight from close prices - add_all is not needed for one average,
    and this pass runs over every file.
    """
    counts: dict[str, list[int]] = {}
    for path in paths:
        try:
            raw = pd.read_csv(path, usecols=["open_time", "close"])
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
        if len(raw) < 400:
            continue
        ema = raw["close"].ewm(span=200, adjust=False).mean()
        above = raw["close"] > ema
        for stamp, is_above in zip(raw["open_time"], above):
            slot = counts.setdefault(stamp, [0, 0])
            slot[0] += 1 if is_above else 0
            slot[1] += 1
    return {t: v[0] / v[1] for t, v in counts.items() if v[1] >= 30}


def daily_slope_map(frame: pd.DataFrame) -> dict:
    """Direction of the daily trend, stamped a day late so nothing leaks.

    A 4h bar inside day D must not see day D's close, so the value carried is
    the previous completed day's slope.
    """
    stamps = pd.to_datetime(frame["time"])
    daily = frame.set_index(stamps)["close"].resample("1D").last().dropna()
    if len(daily) < 60:
        return {}
    ema = daily.ewm(span=50, adjust=False).mean()
    slope = (ema - ema.shift(10)).shift(1)
    return {d.date(): float(v) for d, v in slope.items() if pd.notna(v)}


def specs_from(agents, params=None) -> list[tuple[str, object, dict, bool]]:
    base = dict(C.PARAMS)
    if params:
        base.update(params)
    return [(a.__name__, a, base, a in C.CONTEXT_AGENTS) for a in agents]


def collect(specs: list[tuple[str, object, dict, bool]] | None = None) -> pd.DataFrame:
    """Every base trade, with one vote column per spec.

    specs is [(label, agent, params, needs_context)]. The label is the column
    suffix, so the same agent can be measured at several thresholds.
    """
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_4h.csv")))
    print(f"{len(paths)} symbol files")
    if specs is None:
        specs = specs_from(C.BACKTESTED)

    btc = load(os.path.join(DATA_DIR, "BTCUSDT_4h.csv"))
    if btc is None:
        raise SystemExit("BTCUSDT_4h.csv is needed for the regime agent")
    gap_by_time = dict(
        zip(btc["time"], (btc["close"] - btc["ema200"]) / btc["atr"].replace(0, np.nan))
    )

    breadth = breadth_map(paths)

    rows = []
    for path in paths:
        symbol = os.path.basename(path).replace("_4h.csv", "")
        frame = load(path)
        if frame is None:
            continue

        signals = ema_cross_20_50(frame)
        o = frame["open"].to_numpy(float)
        h = frame["high"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        c = frame["close"].to_numpy(float)
        atr = frame["atr"].to_numpy(float)
        times = frame["time"].tolist()
        slopes = daily_slope_map(frame)

        for i in np.flatnonzero(signals.to_numpy()):
            i = int(i)
            side = int(signals.iloc[i])
            trade = one_trade(o, h, low, c, atr, times, i, side)
            if trade is None:
                continue

            stamp = times[i]
            context = {
                "btc_gap_atr": gap_by_time.get(stamp),
                "breadth": breadth.get(stamp),
                "daily_slope": slopes.get(pd.Timestamp(stamp).date()),
            }
            trade["symbol"] = symbol
            trade["gap"] = context["btc_gap_atr"]
            for label, agent, params, needs_context in specs:
                try:
                    if needs_context:
                        vote = agent(frame, i, side, params, context)
                    else:
                        vote = agent(frame, i, side, params)
                    trade[f"v_{label}"] = vote["vote"]
                except Exception:  # noqa: BLE001 - a broken agent must not stop the run
                    trade[f"v_{label}"] = C.NEUTRAL
            rows.append(trade)

    print(f"{len(rows)} base trades")
    return pd.DataFrame(rows)


def stats(trades: pd.DataFrame, label: str) -> dict:
    if len(trades) == 0:
        return {"label": label, "n": 0}
    r = trades["r"].to_numpy()
    wins = r[r > 0]
    losses = r[r <= 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    return {
        "label": label,
        "n": len(r),
        "win": len(wins) / len(r) * 100,
        "pf": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "exp": r.mean(),
        "total": r.sum(),
    }


def split(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = trades["t"].quantile(IS_FRACTION, interpolation="nearest")
    return trades[trades["t"] <= cut], trades[trades["t"] > cut]


def line(s: dict) -> str:
    if not s["n"]:
        return f"{s['label']:<30} {0:>6}"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (
        f"{s['label']:<30} {s['n']:>6} {s['win']:>6.1f}% {pf:>6} "
        f"{s['exp']:>+7.3f}R {s['total']:>+8.1f}R"
    )


def main() -> int:
    trades = collect()
    if trades.empty:
        print("no trades - check data4h/")
        return 1

    trades["t"] = pd.to_datetime(trades["t"])
    trades = trades.sort_values("t").reset_index(drop=True)
    inside, outside = split(trades)

    header = f"{'committee':<30} {'n':>6} {'win':>7} {'PF':>6} {'expect':>8} {'total':>9}"
    print("\n" + header)
    print("-" * len(header))
    print(line(stats(trades, "base (no committee)")))
    print(line(stats(inside, "  in-sample")))
    print(line(stats(outside, "  out-of-sample")))

    print("\nper agent - the trades it vetoed are removed:")
    for agent in C.BACKTESTED:
        column = f"v_{agent.__name__}"
        if column not in trades:
            continue
        print(line(stats(trades[trades[column] != C.VETO], agent.__name__)))

    print("\nsame agents, out-of-sample only:")
    for agent in C.BACKTESTED:
        column = f"v_{agent.__name__}"
        if column not in trades:
            continue
        print(line(stats(outside[outside[column] != C.VETO], agent.__name__)))

    keep = [a for a in C.BACKTESTED if f"v_{a.__name__}" in trades]
    mask = np.ones(len(trades), dtype=bool)
    for agent in keep:
        mask &= trades[f"v_{agent.__name__}"].to_numpy() != C.VETO
    full = trades[mask]

    print("\nall six together:")
    print(line(stats(full, "all six")))
    if len(full):
        inner, outer = split(full)
        print(line(stats(inner, "  in-sample")))
        print(line(stats(outer, "  out-of-sample")))
        print(f"  kept {len(full) / len(trades) * 100:.1f}% of the base trades")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, "committee_trades.csv")
    trades.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
