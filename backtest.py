"""
Phase 2 - Backtester.

Runs three classic strategies over the saved candles and reports honest numbers:
fees, slippage and funding are subtracted, entries happen at the NEXT bar's open
(never the signal bar's close), and every strategy is judged separately on the
first 70% of history (in-sample) and the last 30% (out-of-sample).

If a strategy only works in-sample, it does not work.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
REPORT_DIR = os.path.join(HERE, "reports")

# --- cost model -------------------------------------------------------------
FEE_RATE = 0.0005      # 0.05% taker, per side
SLIP_RATE = 0.0002     # 0.02% slippage, per side
FUNDING_PER_8H = 0.0001  # ~0.01% average funding, per 8h held

# --- trade model ------------------------------------------------------------
SL_ATR = 2.0           # stop distance = 2 x ATR
RR_RATIO = 2.0         # target = 2R
MAX_HOLD = 30          # bars before we give up (30 x 4h = 5 days)

IS_FRACTION = 0.70     # first 70% of history = in-sample


# --- indicators -------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    df["donchian_high"] = high.rolling(20).max().shift(1)
    df["donchian_low"] = low.rolling(20).min().shift(1)
    df["atr_pct"] = df["atr"] / close

    return df


# --- strategies (signal on bar i -> enter at open of bar i+1) ---------------
def strat_ema_cross(df: pd.DataFrame) -> pd.Series:
    """Trend following: fast EMA crosses slow EMA, filtered by the 200 EMA."""
    fast, slow, trend = df["ema20"], df["ema50"], df["ema200"]
    long_ = (fast > slow) & (fast.shift(1) <= slow.shift(1)) & (df["close"] > trend)
    short = (fast < slow) & (fast.shift(1) >= slow.shift(1)) & (df["close"] < trend)
    return pd.Series(np.where(long_, 1, np.where(short, -1, 0)), index=df.index)


def strat_rsi_pullback(df: pd.DataFrame) -> pd.Series:
    """Buy the dip inside an uptrend, sell the bounce inside a downtrend."""
    uptrend = df["close"] > df["ema200"]
    downtrend = df["close"] < df["ema200"]
    long_ = uptrend & (df["rsi"] < 35) & (df["rsi"].shift(1) >= 35)
    short = downtrend & (df["rsi"] > 65) & (df["rsi"].shift(1) <= 65)
    return pd.Series(np.where(long_, 1, np.where(short, -1, 0)), index=df.index)


def strat_breakout(df: pd.DataFrame) -> pd.Series:
    """Break of the 20-bar range, but only when volatility is not dead."""
    active = df["atr_pct"] > df["atr_pct"].rolling(100).median()
    long_ = (df["close"] > df["donchian_high"]) & active
    short = (df["close"] < df["donchian_low"]) & active
    return pd.Series(np.where(long_, 1, np.where(short, -1, 0)), index=df.index)


STRATEGIES = {
    "ema_cross": strat_ema_cross,
    "rsi_pullback": strat_rsi_pullback,
    "breakout": strat_breakout,
}


# --- trade simulation -------------------------------------------------------
def simulate(df: pd.DataFrame, signal: pd.Series, symbol: str, strategy: str) -> list[dict]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    times = df["open_time"].tolist()
    sig = signal.to_numpy(int)

    trades: list[dict] = []
    i = 1
    n = len(df)

    while i < n - 2:
        side = sig[i]
        if side == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue

        entry = o[i + 1]
        risk = SL_ATR * atr[i]
        if risk <= 0 or entry <= 0:
            i += 1
            continue

        if side > 0:
            stop, target = entry - risk, entry + RR_RATIO * risk
        else:
            stop, target = entry + risk, entry - RR_RATIO * risk

        last = min(i + 1 + MAX_HOLD, n - 1)
        exit_price, exit_idx, reason = None, last, "time"

        for j in range(i + 1, last + 1):
            if side > 0:
                if l[j] <= stop:          # stop first: worst case on purpose
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

        held_bars = exit_idx - i
        gross = (exit_price - entry) / entry * side
        cost = 2 * (FEE_RATE + SLIP_RATE) + FUNDING_PER_8H * (held_bars * 4 / 8)
        net = gross - cost
        risk_pct = risk / entry

        trades.append(
            {
                "symbol": symbol,
                "strategy": strategy,
                "side": "long" if side > 0 else "short",
                "entry_time": times[i + 1],
                "exit_time": times[exit_idx],
                "entry": entry,
                "exit": exit_price,
                "bars": held_bars,
                "reason": reason,
                "gross_pct": gross * 100,
                "net_pct": net * 100,
                "r_multiple": net / risk_pct if risk_pct else 0.0,
            }
        )

        i = exit_idx + 1  # no overlapping trades on the same symbol

    return trades


# --- metrics ----------------------------------------------------------------
def summarise(trades: list[dict], label: str) -> dict:
    if not trades:
        return {"label": label, "trades": 0}

    df = pd.DataFrame(trades).sort_values("entry_time")
    net = df["net_pct"] / 100
    wins, losses = net[net > 0], net[net <= 0]

    equity = (1 + net).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1).min()

    gross_win = wins.sum()
    gross_loss = abs(losses.sum())

    streak = max_streak = 0
    for value in net:
        streak = streak + 1 if value <= 0 else 0
        max_streak = max(max_streak, streak)

    return {
        "label": label,
        "trades": len(df),
        "win_rate": round(len(wins) / len(df) * 100, 1),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
        "expectancy_pct": round(net.mean() * 100, 3),
        "avg_r": round(df["r_multiple"].mean(), 2),
        "total_return_pct": round((equity.iloc[-1] - 1) * 100, 1),
        "max_drawdown_pct": round(drawdown * 100, 1),
        "avg_bars": round(df["bars"].mean(), 1),
        "max_loss_streak": max_streak,
        "stop_hits": int((df["reason"] == "stop").sum()),
        "target_hits": int((df["reason"] == "target").sum()),
    }


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'strategy':<14}{'period':<7}{'trades':>7}{'win%':>7}{'PF':>7}"
        f"{'exp%':>8}{'avgR':>7}{'ret%':>9}{'maxDD%':>8}{'streak':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if not row.get("trades"):
            print(f"{row['label']:<21}no trades")
            continue
        name, _, period = row["label"].partition(" | ")
        print(
            f"{name:<14}{period:<7}{row['trades']:>7}{row['win_rate']:>7}"
            f"{row['profit_factor']:>7}{row['expectancy_pct']:>8}"
            f"{row['avg_r']:>7}{row['total_return_pct']:>9}"
            f"{row['max_drawdown_pct']:>8}{row['max_loss_streak']:>7}"
        )


def main() -> int:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_4h.csv")))
    if not files:
        print("no data found - run: python data.py 4h 730")
        return 1

    os.makedirs(REPORT_DIR, exist_ok=True)
    all_trades: list[dict] = []
    split_time = None

    for path in files:
        symbol = os.path.basename(path).split("_")[0]
        df = pd.read_csv(path)
        df = add_indicators(df)

        if split_time is None:
            ordered = df["open_time"].tolist()
            split_time = ordered[int(len(ordered) * IS_FRACTION)]

        for name, builder in STRATEGIES.items():
            signal = builder(df)
            all_trades.extend(simulate(df, signal, symbol, name))

    print(f"symbols: {len(files)}   total trades: {len(all_trades)}")
    print(f"in-sample ends at: {split_time}\n")

    rows = []
    for name in STRATEGIES:
        subset = [t for t in all_trades if t["strategy"] == name]
        is_part = [t for t in subset if t["entry_time"] < split_time]
        oos_part = [t for t in subset if t["entry_time"] >= split_time]
        rows.append(summarise(subset, f"{name} | all"))
        rows.append(summarise(is_part, f"{name} | IS"))
        rows.append(summarise(oos_part, f"{name} | OOS"))

    print_table(rows)

    detail = pd.DataFrame(all_trades)
    detail.to_csv(os.path.join(REPORT_DIR, "trades.csv"), index=False)

    with open(os.path.join(REPORT_DIR, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "symbols": len(files),
                "cost_model": {
                    "fee_per_side": FEE_RATE,
                    "slippage_per_side": SLIP_RATE,
                    "funding_per_8h": FUNDING_PER_8H,
                },
                "trade_model": {
                    "sl_atr": SL_ATR,
                    "rr": RR_RATIO,
                    "max_hold_bars": MAX_HOLD,
                },
                "split_time": split_time,
                "results": rows,
            },
            handle,
            indent=2,
        )

    print(f"\nreports written to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
