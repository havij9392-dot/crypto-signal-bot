"""
Position sizing and the drawdown brake.

Pure maths on a user dict - no file access, no Telegram. The settings live in
users.py; this module only turns a stop distance into size, leverage and margin.

Sizing logic, in order:
    risk_usd  = capital x risk_pct
    notional  = risk_usd / stop_pct          (so a stop costs exactly risk_usd)
    leverage  = min(MAX_LEVERAGE, 80 / stop_pct)   (never past the liquidation gap)
    margin    = notional / leverage

The brake from riskrules.py is applied to risk_pct: while the account sits more
than 25% below its peak, risk drops to a quarter. That rule beat every other
sizing rule tested on the same 327 trades - higher final balance, smaller
drawdown, and a higher worst point.
"""

from __future__ import annotations

from config import LIQUIDATION_BUFFER, MAX_LEVERAGE, MIN_STOP_PCT

BRAKE_LEVEL = 0.75            # below this share of the peak, risk is cut
BRAKE_FACTOR = 0.25

MAX_RISK_PCT = 10.0
MIN_RISK_PCT = 0.1


def safe_leverage(stop_pct: float) -> int:
    """Highest leverage whose liquidation distance stays beyond the stop."""
    if stop_pct <= 0:
        return 1
    return max(1, min(MAX_LEVERAGE, int(LIQUIDATION_BUFFER / stop_pct)))


def set_capital(user: dict, amount: float) -> None:
    """Explicit new baseline - a deposit or a withdrawal.

    The peak follows the new number, because money the user took out is not a
    trading loss and must not trip the drawdown brake.
    """
    user["capital"] = round(float(amount), 2)
    user["peak"] = user["capital"]


def set_equity(user: dict, amount: float) -> None:
    """Live balance after profit or loss.

    The peak is kept, which is exactly what lets the brake see a real drawdown.
    """
    user["capital"] = round(float(amount), 2)
    user["peak"] = max(float(user.get("peak") or 0.0), user["capital"])


def reset_peak(user: dict) -> None:
    """Start a fresh peak from the current balance."""
    user["peak"] = float(user.get("capital") or 0.0)


def effective_risk(user: dict) -> tuple[float, bool]:
    """Return (risk fraction, brake_active)."""
    base = float(user.get("risk_pct") or 1.0) / 100.0
    peak = float(user.get("peak") or 0.0)
    capital = float(user.get("capital") or 0.0)
    if peak > 0 and capital < peak * BRAKE_LEVEL:
        return base * BRAKE_FACTOR, True
    return base, False


def sizing(capital: float, risk_pct: float, stop_pct: float) -> dict:
    """Turn a stop distance into position size, leverage and margin."""
    if capital <= 0 or stop_pct <= 0:
        return {}

    risk_usd = capital * risk_pct
    notional = risk_usd / (stop_pct / 100.0)
    leverage = safe_leverage(stop_pct)
    margin = notional / leverage

    return {
        "risk_usd": risk_usd,
        "notional": notional,
        "leverage": leverage,
        "margin": margin,
        "margin_pct": margin / capital * 100,
        "loss_pct": min(100.0, stop_pct * leverage),
    }


def describe(user: dict) -> str:
    capital = float(user.get("capital") or 0.0)
    risk_pct, braked = effective_risk(user)
    peak = float(user.get("peak") or 0.0)
    name = user.get("name") or "کاربر"

    if capital <= 0:
        return (
            f"<b>{name}</b>\n"
            "───────────────\n"
            "هنوز سرمایه‌ای ثبت نشده.\n"
            "دکمه «💰 سرمایه» را بزن تا مقدارش را بدهی."
        )

    lines = [
        f"<b>{name}</b>",
        "───────────────",
        f"<b>سرمایه فعلی:</b> ${capital:,.0f}",
        f"<b>بیشترین سرمایه:</b> ${peak:,.0f}",
        f"<b>ریسک هر معامله:</b> {risk_pct * 100:.2f}٪ = ${capital * risk_pct:,.0f}",
    ]
    if braked:
        lines.append("🔻 <b>ترمز افت فعال است</b> — حساب زیر ۷۵٪ اوجش است")
    if peak > capital:
        lines.append(f"<b>افت از اوج:</b> {(capital / peak - 1) * 100:.1f}٪")
    return "\n".join(lines)
