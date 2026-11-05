"""
Can a fixed strategy turn $1000 into $1300 in a month?

Two questions, both answered with data instead of opinion:

  1. Which of the 56 strategies actually made money in the last 30 days?
  2. Is picking last month's winner a real method, or a trap?

Question 2 is the important one. It is tested by walking through every month:
find the best strategy of that month, then measure what it did in the month
after. If the "best of last month" keeps winning, the method works. If it
reverts, the method is overfitting.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from features import add_all
from okx import volumes
from strategies import STRATEGIES
from sweep import EXIT_MODELS, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data4h")
REPORT_DIR = os.path.join(HERE, "reports")

BASKET = 50


def load_basket() -> dict[str, pd.DataFrame]:
    turn = volumes()
    ranked = sorted(
        (
            (os.path.basename(p).split("_")[0], turn.get(os.path.basename(p).split("_")[0][:-4], 0))
            for p in glob.glob(os.path.join(DATA_DIR, "*_4h.csv"))
        ),
        key=lambda item: -item[1],
    )
    frames = {}
    for symbol, _ in ranked[:BASKET]:
        path = os.path.join(DATA_DIR, f"{symbol}_4h.csv")
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path)
        if len(raw) >= 400:
            frames[symbol] = add_all(raw)
    return frames


def all_trades(frames, btc_regime) -> pd.DataFrame:
    rows = []
    for name, builder in STRATEGIES.items():
        for exit_name, model in EXIT_MODELS.items():
            for symbol, df in frames.items():
                signal = builder(df).to_numpy(int)
                regime = np.array([btc_regime.get(t, 0) for t in df["open_time"]])
                signal = np.where(signal == regime, signal, 0)
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
                    rows.append(
                        {
                            "strategy": name,
                            "exit": exit_name,
                            "t": trade["t"],
                            "R": trade["R"],
                            "net": trade["net"],
                        }
                    )
    frame = pd.DataFrame(rows)
    frame["dt"] = pd.to_datetime(frame["t"])
    frame["month"] = frame["dt"].dt.to_period("M")
    return frame


def main() -> int:
    frames = load_basket()
    btc = add_all(pd.read_csv(os.path.join(DATA_DIR, "BTCUSDT_4h.csv")))
    regime = dict(zip(btc["open_time"], np.where(btc["close"] > btc["ema200"], 1, -1)))

    print(f"basket: {len(frames)} symbols · {len(STRATEGIES)} strategies x {len(EXIT_MODELS)} exits")
    print("building trades...")
    trades = all_trades(frames, regime)
    trades["key"] = trades["strategy"] + " | " + trades["exit"]
    print(f"{len(trades):,} trades total\n")

    # ---- 1. what worked in the last 30 days ------------------------------
    end = trades["dt"].max()
    window = trades[trades["dt"] > end - pd.Timedelta(days=30)]

    monthly_last = (
        window.groupby("key")
        .agg(trades=("R", "size"), total_r=("R", "sum"), avg_r=("R", "mean"))
        .sort_values("total_r", ascending=False)
    )

    print("=" * 74)
    print("1. LAST 30 DAYS — what actually made money")
    print("=" * 74)
    print(f"{'strategy':<44}{'n':>6}{'total R':>10}{'avg R':>9}")
    print("-" * 74)
    for key, row in monthly_last.head(12).iterrows():
        print(f"{key:<44}{int(row['trades']):>6}{row['total_r']:>+10.1f}{row['avg_r']:>+9.3f}")
    print(f"\nprofitable in this month: {int((monthly_last['total_r'] > 0).sum())} of {len(monthly_last)}")

    # ---- 2. does last month's winner win next month? ---------------------
    per_month = trades.groupby(["month", "key"])["R"].sum().unstack(fill_value=0.0)
    per_month = per_month.loc[per_month.index.sort_values()]
    # drop the current partial month
    per_month = per_month.iloc[:-1]
    months = list(per_month.index)

    print()
    print("=" * 74)
    print("2. PICK LAST MONTH'S WINNER — does it repeat?")
    print("=" * 74)
    print(f"{'month':<10}{'winner':<40}{'that month':>11}{'next month':>12}")
    print("-" * 74)

    picked, following, all_avg = [], [], []
    for i in range(len(months) - 1):
        current, nxt = months[i], months[i + 1]
        row = per_month.loc[current]
        winner = row.idxmax()
        this_r = float(row[winner])
        next_r = float(per_month.loc[nxt, winner])
        picked.append(this_r)
        following.append(next_r)
        all_avg.append(float(per_month.loc[nxt].mean()))
        if i >= len(months) - 13:
            print(f"{str(current):<10}{winner:<40}{this_r:>+11.1f}{next_r:>+12.1f}")

    picked = np.array(picked)
    following = np.array(following)
    all_avg = np.array(all_avg)

    print("-" * 74)
    print(f"average of the picked strategy in the month it was picked : {picked.mean():+.1f}R")
    print(f"average of that same strategy in the FOLLOWING month     : {following.mean():+.1f}R")
    print(f"average of just taking every strategy's next month        : {all_avg.mean():+.1f}R")
    print(f"share of picks that were profitable the next month        : {(following > 0).mean() * 100:.0f}%")

    # ---- 3. the fixed strategy, month by month ---------------------------
    chosen = "ema_cross_20_50 | fixed_3r"
    print()
    print("=" * 74)
    print(f"3. THE FIXED STRATEGY, MONTH BY MONTH  ({chosen})")
    print("=" * 74)
    print(f"{'month':<10}{'trades':>7}{'total R':>10}{'@0.5%':>9}{'@1%':>9}{'@2%':>9}")
    print("-" * 74)
    fixed = trades[trades["key"] == chosen]
    for month, group in fixed.groupby("month"):
        total = group["R"].sum()
        print(
            f"{str(month):<10}{len(group):>7}{total:>+10.1f}"
            f"{(total * 0.005 * 100):>+8.1f}%{(total * 0.01 * 100):>+8.1f}%{(total * 0.02 * 100):>+8.1f}%"
        )

    per_month_r = fixed.groupby("month")["R"].sum()
    print("-" * 74)
    print(f"average month: {per_month_r.mean():+.1f}R   best: {per_month_r.max():+.1f}R   worst: {per_month_r.min():+.1f}R")
    for risk in (0.005, 0.01, 0.02, 0.03):
        rets = per_month_r * risk
        hit = (rets >= 0.30).sum()
        print(
            f"  risk {risk * 100:.1f}%: {hit}/{len(rets)} months reached +30% "
            f"· average {rets.mean() * 100:+.1f}% · worst {rets.min() * 100:+.1f}%"
        )

    os.makedirs(REPORT_DIR, exist_ok=True)
    monthly_last.to_csv(os.path.join(REPORT_DIR, "last_month_ranking.csv"))
    per_month.to_csv(os.path.join(REPORT_DIR, "monthly_by_strategy.csv"))
    print(f"\nwritten to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
