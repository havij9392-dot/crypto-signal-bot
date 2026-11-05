"""
Account settings and position sizing.

The bot asks Amir how much capital he is starting with, stores it in
state/account.json, and every signal is then sized from that number instead of
showing percentages only.

Sizing logic, in order:
    risk_usd  = capital x risk_pct
    notional  = risk_usd / stop_pct          (so a stop costs exactly risk_usd)
    leverage  = min(MAX_LEVERAGE, 80 / stop_pct)   (never past the liquidation gap)
    margin    = notional / leverage

The drawdown brake from riskrules.py is applied to risk_pct: while the account
sits more than 25% below its peak, risk drops to a quarter.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, "state")
ACCOUNT_FILE = os.path.join(STATE_DIR, "account.json")

MAX_LEVERAGE = 10
LIQUIDATION_BUFFER = 80.0     # percent of the 100/leverage gap we are willing to use
BRAKE_LEVEL = 0.75            # below this share of the peak, risk is cut
BRAKE_FACTOR = 0.25

DEFAULTS = {
    "capital": 0.0,           # 0 means "not set yet", the bot asks for it
    "risk_pct": 1.0,
    "peak": 0.0,
    "last_update_id": 0,
    "awaiting": None,
}


def load() -> dict:
    data = dict(DEFAULTS)
    if os.path.exists(ACCOUNT_FILE):
        try:
            with open(ACCOUNT_FILE, encoding="utf-8") as handle:
                data.update(json.load(handle))
        except (json.JSONDecodeError, OSError):
            pass
    return data


def save(data: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(ACCOUNT_FILE, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def set_capital(data: dict, amount: float) -> None:
    """Explicit new baseline - a deposit or a withdrawal.

    The peak follows the new number, because money the user took out is not a
    trading loss and must not trip the drawdown brake.
    """
    data["capital"] = round(float(amount), 2)
    data["peak"] = data["capital"]


def set_equity(data: dict, amount: float) -> None:
    """Live balance after profit or loss.

    The peak is kept, which is exactly what lets the brake see a real drawdown.
    """
    data["capital"] = round(float(amount), 2)
    data["peak"] = max(float(data.get("peak") or 0.0), data["capital"])


def reset_peak(data: dict) -> None:
    """Start a fresh peak from the current balance."""
    data["peak"] = float(data.get("capital") or 0.0)


def effective_risk(data: dict) -> tuple[float, bool]:
    """Return (risk fraction, brake_active)."""
    base = float(data.get("risk_pct") or 1.0) / 100.0
    peak = float(data.get("peak") or 0.0)
    capital = float(data.get("capital") or 0.0)
    if peak > 0 and capital < peak * BRAKE_LEVEL:
        return base * BRAKE_FACTOR, True
    return base, False


def sizing(capital: float, risk_pct: float, stop_pct: float) -> dict:
    """Turn a stop distance into position size, leverage and margin."""
    if capital <= 0 or stop_pct <= 0:
        return {}

    risk_usd = capital * risk_pct
    notional = risk_usd / (stop_pct / 100.0)
    leverage = max(1, min(MAX_LEVERAGE, int(LIQUIDATION_BUFFER / stop_pct)))
    margin = notional / leverage

    return {
        "risk_usd": risk_usd,
        "notional": notional,
        "leverage": leverage,
        "margin": margin,
        "margin_pct": margin / capital * 100,
        "loss_pct": min(100.0, stop_pct * leverage),
    }


def describe(data: dict) -> str:
    capital = float(data.get("capital") or 0.0)
    risk_pct, braked = effective_risk(data)
    peak = float(data.get("peak") or 0.0)

    if capital <= 0:
        return (
            "هنوز سرمایه‌ای ثبت نشده.\n"
            "دستور /capital را بزن تا مقدار را بدهی."
        )

    lines = [
        "<b>تنظیمات حساب</b>",
        "───────────────",
        f"<b>سرمایه فعلی:</b> ${capital:,.0f}",
        f"<b>بیشترین سرمایه:</b> ${peak:,.0f}",
        f"<b>ریسک هر معامله:</b> {risk_pct * 100:.2f}٪ = ${capital * risk_pct:,.0f}",
    ]
    if braked:
        lines.append("🔻 <b>ترمز افت فعال است</b> — حساب زیر ۷۵٪ اوجش است")
    if peak > capital:
        lines.append(f"<b>افت از اوج:</b> {(capital / peak - 1) * 100:.1f}٪")
    lines.append("───────────────")
    lines.append("/capital برای تغییر سرمایه · /equity برای به‌روز کردن موجودی")
    return "\n".join(lines)
