"""
One place for the numbers that define the strategy.

Before this file the same constants lived in both scanner.py and account.py with
slightly different units (0.80 vs 80.0), which is exactly how a sizing bug gets
in. Everything now imports from here.

These values are not free parameters - they are the ones the 56-strategy sweep
and the 5-month basket test were run with (reports/strategy_winner.md).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

INTERVAL = "4h"
LOOKBACK_DAYS = 200        # 1200 bars: comfortably past the 200 EMA

SL_ATR = 2.5               # stop distance in ATR
RR = 3.0                   # target as a multiple of the stop
MAX_HOLD_BARS = 60         # give up after 10 days on 4h

MIN_TURNOVER_USD = 3_000_000
MIN_STOP_PCT = 1.5         # below this the fee eats the edge
WORKERS = 10

MAX_LEVERAGE = 10          # hard ceiling chosen by Amir
LIQUIDATION_BUFFER = 80.0  # percent of the 100/leverage gap we are willing to use

PAPER_BANNER = "⚠️ حالت آزمایشی — این سیگنال هنوز با پول واقعی تأیید نشده"

# ------------------------------------------------------------------ زمان‌ها
# همه‌چیز داخل کد با UTC است، چون صرافی و GitHub با UTC کار می‌کنند و مقایسه
# کندل‌ها باید یک مرجع داشته باشد. این فقط برای نمایش است.
TEHRAN = timezone(timedelta(hours=3, minutes=30), "تهران")

# candle stamps are "2026-11-05 16:00", user records are ISO with a Z
_STAMP_FORMATS = (
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
)


def to_tehran(stamp: str, add_hours: int = 0) -> str:
    """'2026-11-05 16:00' (UTC) -> '2026-11-05 19:30' به وقت تهران."""
    if not stamp:
        return stamp
    moment = None
    for fmt in _STAMP_FORMATS:
        try:
            moment = datetime.strptime(str(stamp), fmt).replace(tzinfo=timezone.utc)
            break
        except (ValueError, TypeError):
            continue
    if moment is None:
        return stamp
    return (moment + timedelta(hours=add_hours)).astimezone(TEHRAN).strftime(
        "%Y-%m-%d %H:%M"
    )
