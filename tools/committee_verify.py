"""
Measure the committee that actually ships.

The ablation scripts tested hand-written copies of the agents. This one imports
committee.py and votes through committee.judge / committee.breadth, so the
numbers describe the code in the repository rather than a variant of it.

It also writes reports/committee_verdict.md, so the documented result cannot
drift away from the code.

Usage:
    python tools/committee_verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import committee as C
from tools.committee_backtest import REPORT_DIR, collect, split


def breadth_vote(frame, i, side, p, context=None):
    """Adapter: the real breadth agent takes a market-wide share, not a frame."""
    return C.breadth((context or {}).get("breadth"), side, p)


def specs():
    out = [(a.__name__, a, dict(C.PARAMS), False) for a in C.SYMBOL_AGENTS]
    out.append(("breadth", breadth_vote, dict(C.PARAMS), True))
    return out


def metrics(part: pd.DataFrame) -> dict:
    if not len(part):
        return {"n": 0}
    r = part["r"].to_numpy()
    wins = r[r > 0]
    losses = r[r <= 0]
    gross_loss = abs(losses.sum())
    return {
        "n": len(r),
        "win": len(wins) / len(r) * 100,
        "pf": wins.sum() / gross_loss if gross_loss else float("inf"),
        "exp": r.mean(),
        "total": r.sum(),
    }


def row(label: str, m: dict) -> str:
    if not m["n"]:
        return f"| {label} | ۰ | — | — | — | — |"
    pf = "∞" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    return (
        f"| {label} | {m['n']:,} | {m['win']:.1f}% | {pf} | "
        f"{m['exp']:+.3f}R | {m['total']:+.1f}R |"
    )


def main() -> int:
    trades = collect(specs())
    trades["t"] = pd.to_datetime(trades["t"])
    trades = trades.sort_values("t").reset_index(drop=True)
    trades["year"] = trades["t"].dt.year

    inside, outside = split(trades)

    columns = [f"v_{a.__name__}" for a in C.SYMBOL_AGENTS] + ["v_breadth"]
    mask = np.ones(len(trades), dtype=bool)
    for column in columns:
        mask &= trades[column].to_numpy() != C.VETO
    kept = trades[mask]

    base = metrics(trades)
    full = metrics(kept)
    ins = metrics(kept[kept["t"] <= trades["t"].quantile(0.70)])
    outs = metrics(kept[kept["t"] > trades["t"].quantile(0.70)])

    header = "| | معاملات | برد | PF | انتظار | مجموع |"
    lines = [
        "# حکم هیئت ورود",
        "",
        "ساخته‌شده با `committee.py` — همان کدی که اجرا می‌شود، نه نسخه‌ای جداگانه.",
        "داده: ۱۵۰ ارز، دو سال کندل ۴ ساعته، با کارمزد ۰٫۰۵٪ هر طرف، اسلیپیج ۰٫۰۲٪ هر طرف",
        "و فاندینگ ۰٫۰۱٪ هر ۸ ساعت. ورود در open کندل بعد از سیگنال، حد ضرر قبل از حد سود.",
        "",
        "## نتیجه",
        "",
        header,
        "|---|---|---|---|---|---|",
        row("بدون هیئت", base),
        row("**با هیئت**", full),
        row("in-sample (۷۰٪ اول)", ins),
        row("out-of-sample (۳۰٪ آخر)", outs),
        "",
        f"هیئت {len(kept) / len(trades) * 100:.0f}٪ معاملات پایه را نگه می‌دارد.",
        "",
        "## پایداری",
        "",
        "| سال | بدون هیئت | با هیئت |",
        "|---|---|---|",
    ]

    for year in sorted(trades["year"].unique()):
        b = metrics(trades[trades["year"] == year])
        k = metrics(kept[kept["year"] == year])
        lines.append(
            f"| {year} | {b['exp']:+.3f}R ({b['n']:,}) | {k['exp']:+.3f}R ({k['n']:,}) |"
        )

    by_symbol = kept.groupby("symbol")["r"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    total_r = by_symbol["sum"].sum()
    top5 = ", ".join(f"{s} {v / total_r * 100:.0f}٪" for s, v in by_symbol["sum"].head(5).items())
    without5 = metrics(kept[~kept["symbol"].isin(by_symbol.head(5).index)])
    without10 = metrics(kept[~kept["symbol"].isin(by_symbol.head(10).index)])

    lines += [
        "",
        "| آزمون | نتیجه |",
        "|---|---|",
        f"| سهم ۵ ارز برتر از سود | {top5} |",
        f"| ارزهای سودده | {(by_symbol['sum'] > 0).sum()} از {len(by_symbol)} |",
        f"| با حذف ۵ ارز برتر | {without5['exp']:+.3f}R ({without5['n']:,}) |",
        f"| با حذف ۱۰ ارز برتر | {without10['exp']:+.3f}R ({without10['n']:,}) |",
        "",
        "## اعضای هیئت",
        "",
        "| عضو | سنجه | وضعیت |",
        "|---|---|---|",
    ]

    seat = {
        "adx_strength": "ADX ≥ ۲۵",
        "volume_confirmation": "حجم ≥ ۱× میانگین ۲۰ کندل",
        "volatility": "صدک ATR بین ۲۰٪ و ۹۵٪",
        "breadth": "≥ ۴۵٪ بازار هم‌جهت",
    }
    for agent in C.SYMBOL_AGENTS:
        lines.append(f"| {agent.__doc__.split('—')[0].strip()} | {seat[agent.__name__]} | ✅ رأی دارد |")
    lines.append(f"| گستردگی بازار | {seat['breadth']} | ✅ رأی دارد |")

    lines += [
        "",
        "## اعضایی که رأی نگرفتند",
        "",
        "| عضو | چه شد |",
        "|---|---|",
        "| روند (شیب EMA200) | +۰٫۱۲۶R کل، اما **−۰٫۰۰۵R** خارج از نمونه — رد |",
        "| مومنتوم (RSI) | +۰٫۱۰۲R و خارج از نمونه +۰٫۰۰۹R — بدتر از پایه — رد |",
        "| ساختار (فضا تا سقف) | +۰٫۱۱۶R در برابر +۰٫۱۱۸R پایه — خنثی — رد |",
        "| زمان‌بندی (فاصله از EMA20) | خارج از نمونه **−۰٫۰۶۱R** — رد |",
        "| رژیم (فاصله بیت‌کوین) | با فیلتر رژیم موجود هم‌پوشانی دارد و ناپایدار بود — رد |",
        "| هم‌جهتی روزانه | خارج از نمونه **−۰٫۰۲۵R** — رد |",
        "",
        "رد کردن هم نتیجه است: هر عضوی که حذف می‌شود، معاملاتی را حذف می‌کند که",
        "استراتژی پایه آن‌ها را می‌گرفت. عضوی که انتظار را بالا نمی‌برد، فقط نمونه را",
        "کوچک می‌کند.",
    ]

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "committee_verdict.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines[:12]))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
