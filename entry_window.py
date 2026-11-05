"""
Phase 5 - How wide should the entry window be?

The live scanner only fires on the exact 30m candle where EMA20 crosses EMA50.
That is a one-candle window: if the scan misses it, the setup is gone forever.
This module measures what happens if we allow the cross to be up to N candles
old, as long as price is still on the right side of the 200 EMA and no opposite
cross happened in between.

Same cost model and same fill assumptions as backtest.py - nothing is relaxed.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from backtest import FEE_RATE, FUNDING_PER_8H, SLIP_RATE, add_indicators

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

BAR_MINUTES = 30
SL_ATR = 2.5
RR = 3.0
MAX_HOLD = 96          # 96 x 30m = 48 hours
WINDOWS = [1, 2, 3, 4, 6, 8, 12, 24]
IS_FRACTION = 0.70


def load_30m(path: str) -> pd.DataFrame:
    """Resample the saved 15m candles up to 30m, then add indicators."""
    raw = pd.read_csv(path)
    raw["ts"] = pd.to_datetime(raw["open_time"], utc=True)
    raw = raw.set_index("ts")

    agg = raw.resample(f"{BAR_MINUTES}min", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg = agg.dropna(subset=["open", "close"]).reset_index()
    agg = agg.rename(columns={"ts": "ts"})
    agg["open_time"] = agg["ts"].dt.strftime("%Y-%m-%d %H:%M")
    return add_indicators(agg)


def btc_regime_series(ts: pd.Series) -> pd.Series:
    """BTC above/below its 4h EMA200, aligned to each 30m candle close."""
    path = os.path.join(DATA_DIR, "BTCUSDT_4h.csv")
    btc = pd.read_csv(path)
    btc["ts"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = add_indicators(btc)
    btc["regime"] = np.where(btc["close"] > btc["ema200"], 1, -1)

    right = btc[["ts", "regime"]].sort_values("ts")
    left = pd.DataFrame({"ts": ts}).sort_values("ts")
    merged = pd.merge_asof(left, right, on="ts", direction="backward")
    merged = merged.set_index(left.index).sort_index()
    return merged["regime"].fillna(-1).astype(int)


def window_signal(df: pd.DataFrame, window: int) -> pd.Series:
    """+1/-1 while a cross that happened up to `window` bars ago is still valid."""
    fast, slow, trend, close = df["ema20"], df["ema50"], df["ema200"], df["close"]

    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    # bar offset (0 = this bar) of the most recent cross in each direction
    idx = pd.Series(np.arange(len(df)), index=df.index)
    up_at = idx.where(cross_up).ffill()
    dn_at = idx.where(cross_dn).ffill()

    age_up = idx - up_at
    age_dn = idx - dn_at

    long_ = (age_up < window) & (close > trend) & (age_up < age_dn)
    short = (age_dn < window) & (close < trend) & (age_dn < age_up)

    return pd.Series(np.where(long_, 1, np.where(short, -1, 0)), index=df.index)


def simulate(df: pd.DataFrame, signal: pd.Series, regime: pd.Series) -> list[dict]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low_arr = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    times = df["open_time"].tolist()
    sig = signal.to_numpy(int)
    reg = regime.to_numpy(int)

    trades: list[dict] = []
    i = 1
    n = len(df)

    while i < n - 2:
        side = sig[i]
        if side == 0 or side != reg[i] or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue

        entry = o[i + 1]
        risk = SL_ATR * atr[i]
        if risk <= 0 or entry <= 0:
            i += 1
            continue

        if side > 0:
            stop, target = entry - risk, entry + RR * risk
        else:
            stop, target = entry + risk, entry - RR * risk

        last = min(i + 1 + MAX_HOLD, n - 1)
        exit_price, exit_idx = None, last

        for j in range(i + 1, last + 1):
            if side > 0:
                if low_arr[j] <= stop:
                    exit_price, exit_idx = stop, j
                    break
                if h[j] >= target:
                    exit_price, exit_idx = target, j
                    break
            else:
                if h[j] >= stop:
                    exit_price, exit_idx = stop, j
                    break
                if low_arr[j] <= target:
                    exit_price, exit_idx = target, j
                    break

        if exit_price is None:
            exit_price = c[last]

        held = exit_idx - i
        gross = (exit_price - entry) / entry * side
        cost = 2 * (FEE_RATE + SLIP_RATE) + FUNDING_PER_8H * (held * BAR_MINUTES / 480)
        net = gross - cost
        risk_pct = risk / entry

        trades.append(
            {
                "entry_time": times[i + 1],
                "net_pct": net * 100,
                "r_multiple": net / risk_pct if risk_pct else 0.0,
            }
        )
        i = exit_idx + 1

    return trades


def stats(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "win": 0.0, "pf": 0.0, "avg_r": 0.0, "ret": 0.0, "dd": 0.0}
    frame = pd.DataFrame(trades).sort_values("entry_time")
    net = frame["net_pct"] / 100
    wins, losses = net[net > 0], net[net <= 0]
    equity = (1 + net).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    gl = abs(losses.sum())
    return {
        "trades": len(frame),
        "win": round(len(wins) / len(frame) * 100, 1),
        "pf": round(wins.sum() / gl, 2) if gl else 0.0,
        "avg_r": round(frame["r_multiple"].mean(), 3),
        "ret": round((equity.iloc[-1] - 1) * 100, 1),
        "dd": round(dd * 100, 1),
    }


def main() -> int:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_15m.csv")))
    if not files:
        print("no 15m data - run: python data.py 15m 365")
        return 1

    frames: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for path in files:
        symbol = os.path.basename(path).split("_")[0]
        df = load_30m(path)
        frames[symbol] = (df, btc_regime_series(df["ts"]))

    print(f"symbols: {len(frames)}   bars per symbol: ~{len(next(iter(frames.values()))[0])}")
    print(f"range: {next(iter(frames.values()))[0]['open_time'].iloc[0]} -> "
          f"{next(iter(frames.values()))[0]['open_time'].iloc[-1]}\n")

    all_times = sorted(
        {t for df, _ in frames.values() for t in df["open_time"].tolist()}
    )
    split = all_times[int(len(all_times) * IS_FRACTION)]

    header = (
        f"{'window':<8}{'trades':>7}{'win%':>7}{'PF':>7}{'avgR':>8}"
        f"{'ret%':>9}{'maxDD%':>8}   {'IS_PF':>7}{'OOS_PF':>8}{'OOS_R':>8}"
    )
    print(header)
    print("-" * len(header))

    for window in WINDOWS:
        trades: list[dict] = []
        for symbol, (df, regime) in frames.items():
            trades.extend(simulate(df, window_signal(df, window), regime))

        is_part = [t for t in trades if t["entry_time"] < split]
        oos_part = [t for t in trades if t["entry_time"] >= split]

        overall = stats(trades)
        is_s = stats(is_part)
        oos_s = stats(oos_part)

        print(
            f"{window:<8}{overall['trades']:>7}{overall['win']:>7}{overall['pf']:>7}"
            f"{overall['avg_r']:>8}{overall['ret']:>9}{overall['dd']:>8}   "
            f"{is_s['pf']:>7}{oos_s['pf']:>8}{oos_s['avg_r']:>8}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
