"""
Which agents deserve a seat?

The first six agents were chosen because they sound right. This treats them as
hypotheses and tests a wider field on the same trades: ADX, volume, market
breadth, daily alignment, and several threshold variants of the volatility
filter that already showed promise.

The rule for keeping an agent: expectancy must beat the base strategy in the
full period AND in the out-of-sample half. An agent that only helps in-sample
is an artefact; an agent that removes trades without improving what is left is
just a smaller sample.

Usage:
    python tools/committee_explore.py
"""

from __future__ import annotations

import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from committee import APPROVE, VETO, _f, _neutral, _vote
from tools.committee_backtest import collect, split


# ------------------------------------------------------------ candidate agents


def adx(frame, i, side, p):
    """قدرت روند — a cross in a trendless market has nothing to follow."""
    value = _f(frame, "adx", i)
    if not math.isfinite(value):
        return _neutral("adx", "قدرت روند", "ADX نامعتبر")
    if value < p["adx_min"]:
        return _vote("adx", "قدرت روند", VETO, f"ADX {value:.0f} — روند ضعیف")
    return _vote("adx", "قدرت روند", APPROVE, f"ADX {value:.0f}")


def volume(frame, i, side, p):
    """تأیید حجم — a cross nobody traded is usually a fake-out."""
    vol = _f(frame, "volume", i)
    avg = _f(frame, "vol_sma20", i)
    if not math.isfinite(vol) or not math.isfinite(avg) or avg <= 0:
        return _neutral("volume", "حجم", "حجم نامعتبر")
    ratio = vol / avg
    if ratio < p["volume_min"]:
        return _vote("volume", "حجم", VETO, f"حجم {ratio:.1f}× میانگین")
    return _vote("volume", "حجم", APPROVE, f"حجم {ratio:.1f}× میانگین")


def breadth(frame, i, side, p, context=None):
    """گستردگی بازار — is the whole market moving, or is this one coin alone?"""
    share = (context or {}).get("breadth")
    if share is None or not math.isfinite(float(share)):
        return _neutral("breadth", "گستردگی", "داده بازار نیست")
    share = float(share)
    aligned = share if side > 0 else 1.0 - share
    if aligned < p["breadth_min"]:
        return _vote("breadth", "گستردگی", VETO, f"فقط {share * 100:.0f}٪ بازار هم‌جهت")
    return _vote("breadth", "گستردگی", APPROVE, f"{share * 100:.0f}٪ بازار هم‌جهت")


def daily(frame, i, side, p, context=None):
    """هم‌جهتی روزانه — a 4h cross should not fight the daily trend."""
    slope = (context or {}).get("daily_slope")
    if slope is None or not math.isfinite(float(slope)):
        return _neutral("daily", "هم‌جهتی روزانه", "داده روزانه نیست")
    aligned = float(slope) if side > 0 else -float(slope)
    if aligned < 0:
        return _vote("daily", "هم‌جهتی روزانه", VETO, "روند روزانه مخالف است")
    return _vote("daily", "هم‌جهتی روزانه", APPROVE, "روند روزانه هم‌جهت")


def volatility_variant(frame, i, side, p):
    """نوسان با آستانه قابل تغییر — the agent that already showed promise."""
    rank = _f(frame, "atr_rank", i)
    if not math.isfinite(rank):
        return _neutral("vol", "نوسان", "رتبه ATR نامعتبر")
    if rank < p["atr_rank_lo"]:
        return _vote("vol", "نوسان", VETO, f"بازار مرده (صدک {rank * 100:.0f})")
    if rank > p["atr_rank_hi"]:
        return _vote("vol", "نوسان", VETO, f"نوسان بی‌ثبات (صدک {rank * 100:.0f})")
    return _vote("vol", "نوسان", APPROVE, f"نوسان نرمال (صدک {rank * 100:.0f})")


VARIANTS = [
    ("adx>=18", adx, {"adx_min": 18}),
    ("adx>=22", adx, {"adx_min": 22}),
    ("adx>=25", adx, {"adx_min": 25}),
    ("vol>=0.8x", volume, {"volume_min": 0.8}),
    ("vol>=1.0x", volume, {"volume_min": 1.0}),
    ("vol>=1.3x", volume, {"volume_min": 1.3}),
    ("breadth>=45", breadth, {"breadth_min": 0.45}),
    ("breadth>=55", breadth, {"breadth_min": 0.55}),
    ("breadth>=65", breadth, {"breadth_min": 0.65}),
    ("daily", daily, {}),
    ("atr 0.05-0.98", volatility_variant, {"atr_rank_lo": 0.05, "atr_rank_hi": 0.98}),
    ("atr 0.10-0.95", volatility_variant, {"atr_rank_lo": 0.10, "atr_rank_hi": 0.95}),
    ("atr 0.20-0.95", volatility_variant, {"atr_rank_lo": 0.20, "atr_rank_hi": 0.95}),
    ("atr 0.15-0.90", volatility_variant, {"atr_rank_lo": 0.15, "atr_rank_hi": 0.90}),
    ("atr 0.15-0.99", volatility_variant, {"atr_rank_lo": 0.15, "atr_rank_hi": 0.99}),
]


def fmt(part: pd.DataFrame) -> str:
    if not len(part):
        return "     -"
    r = part["r"].to_numpy()
    wins = r[r > 0]
    losses = abs(r[r <= 0].sum())
    pf = wins.sum() / losses if losses else float("inf")
    pf = "inf" if pf == float("inf") else f"{pf:.2f}"
    return f"{len(r):>5} {r.mean():>+7.3f}R {pf:>5}"


def main() -> int:
    specs = [(label, agent, dict(p), agent is breadth or agent is daily)
             for label, agent, p in VARIANTS]
    trades = collect(specs)
    trades["t"] = pd.to_datetime(trades["t"])
    trades = trades.sort_values("t").reset_index(drop=True)
    inside, outside = split(trades)
    inside_t, outside_t = set(inside["t"]), set(outside["t"])

    print(f"\n{'agent / variant':<16} {'all trades':>20} | {'in-sample':>20} | {'out-of-sample':>20}")
    print("-" * 84)

    def row(label, kept):
        ins = kept[kept["t"].isin(inside_t)]
        out = kept[kept["t"].isin(outside_t)]
        print(f"{label:<16} {fmt(kept):>20} | {fmt(ins):>20} | {fmt(out):>20}")

    row("base", trades)
    for label, agent, p in VARIANTS:
        column = f"v_{label}"
        if column not in trades:
            continue
        row(label, trades[trades[column] != VETO])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
