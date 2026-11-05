"""
Paper-trading book.

Every signal the scanner sends is written down here and then followed forward on
the real candles until it stops out, hits target, or runs out of time. The point
is a track record nobody can edit after the fact: the entry, the stop and the
target are fixed at signal time, exactly like the backtest.

The book is global, not per user - it measures the strategy, not anybody's
account. Each user's own money is derived from these R values through
account.sizing().

R maths, matching backtest.py:
    risk_price = entry x stop_pct / 100
    long  : R = (exit - entry) / risk_price
    short : R = (entry - exit) / risk_price

The stop is always checked before the target inside the same candle, because a
4h candle does not tell us which came first. That is the pessimistic assumption,
same as the backtest.

Usage:
    python papertrade.py          # update the book and print the result
    python papertrade.py --stats  # just show the numbers
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from config import MAX_HOLD_BARS, to_tehran
from okx import fetch_klines

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, "state")
BOOK_FILE = os.path.join(STATE_DIR, "paper.json")

MAX_CLOSED = 500
LOOKBACK_DAYS = 60          # enough for a 10-day trade plus slack


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def when_tehran(stamp: str | None) -> str:
    """Candle stamps are UTC at the bar's open, so +4h lands on its close."""
    if not stamp:
        return "?"
    text = to_tehran(stamp, 4)
    return text[5:] if len(text) >= 16 else text


def load() -> dict:
    if not os.path.exists(BOOK_FILE):
        return {"open": [], "closed": []}
    try:
        with open(BOOK_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {"open": [], "closed": []}
    data.setdefault("open", [])
    data.setdefault("closed", [])
    return data


def save(book: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    book["closed"] = book["closed"][-MAX_CLOSED:]
    tmp = BOOK_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(book, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, BOOK_FILE)


def trade_id(symbol: str, candle: str) -> str:
    return f"{symbol}@{candle}"


def has_open(book: dict, symbol: str) -> bool:
    return any(trade["symbol"] == symbol for trade in book["open"])


def record(book: dict, plan: dict, recipients: dict | None = None) -> dict | None:
    """Write a new signal down. Returns None if the symbol already has one open.

    recipients is {chat_id: {"message_id": id, "risk_usd": dollars}} - the anchor
    the outcome will be replied under, plus the money that user had at risk. The
    book is still global; this is only so the reply can be personal.
    """
    if has_open(book, plan["symbol"]):
        return None

    trade = {
        "id": trade_id(plan["symbol"], plan["candle"]),
        "symbol": plan["symbol"],
        "side": int(plan["side"]),
        "entry": float(plan["price"]),
        "stop_pct": float(plan["stop_pct"]),
        "target_pct": float(plan["target_pct"]),
        "opened": plan["candle"],
        "opened_at": _now(),
        "recipients": dict(recipients or {}),
    }
    book["open"].append(trade)
    return trade


def _close(trade: dict, exit_price: float, when: str, reason: str) -> dict:
    risk_price = trade["entry"] * trade["stop_pct"] / 100.0
    if risk_price <= 0:
        r = 0.0
    elif trade["side"] > 0:
        r = (exit_price - trade["entry"]) / risk_price
    else:
        r = (trade["entry"] - exit_price) / risk_price

    closed = dict(trade)
    closed.update(
        {
            "exit": float(exit_price),
            "closed": when,
            "closed_at": _now(),
            "reason": reason,
            "r": round(r, 4),
        }
    )
    return closed


def walk(trade: dict, candles: list[list]) -> dict | None:
    """Follow one trade through the candles after it opened.

    candles are OKX rows: [ts, open, high, low, close, ...]. Returns a closed
    trade, or None while it is still running.
    """
    opened = trade["opened"]
    target_mult = trade["target_pct"] / trade["stop_pct"]

    risk_price = trade["entry"] * trade["stop_pct"] / 100.0
    if trade["side"] > 0:
        stop = trade["entry"] - risk_price
        target = trade["entry"] + risk_price * target_mult
    else:
        stop = trade["entry"] + risk_price
        target = trade["entry"] - risk_price * target_mult

    seen = 0
    last_close = None
    last_time = None

    for row in candles:
        when = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        if when <= opened:
            continue

        high, low, close = float(row[2]), float(row[3]), float(row[4])
        seen += 1
        last_close, last_time = close, when

        if trade["side"] > 0:
            if low <= stop:
                return _close(trade, stop, when, "stop")
            if high >= target:
                return _close(trade, target, when, "target")
        else:
            if high >= stop:
                return _close(trade, stop, when, "stop")
            if low <= target:
                return _close(trade, target, when, "target")

        if seen >= MAX_HOLD_BARS:
            return _close(trade, close, when, "timeout")

    return None


def update(book: dict, fetcher=fetch_klines) -> list[dict]:
    """Move every open trade forward. Returns the trades that closed."""
    still_open: list[dict] = []
    closed_now: list[dict] = []

    for trade in book["open"]:
        try:
            candles = fetcher(trade["symbol"], "4h", LOOKBACK_DAYS)
        except Exception as exc:  # noqa: BLE001 - a dead symbol must not stop the book
            print(f"  {trade['symbol']:<12} could not refresh ({exc}), keeping it open")
            still_open.append(trade)
            continue

        result = walk(trade, candles)
        if result is None:
            still_open.append(trade)
        else:
            closed_now.append(result)
            book["closed"].append(result)

    book["open"] = still_open
    return closed_now


def stats(book: dict) -> dict:
    closed = book.get("closed", [])
    wins = [t for t in closed if t["r"] > 0]
    losses = [t for t in closed if t["r"] <= 0]
    gross_win = sum(t["r"] for t in wins)
    gross_loss = abs(sum(t["r"] for t in losses))

    return {
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(closed) * 100) if closed else 0.0,
        "sum_r": sum(t["r"] for t in closed),
        "avg_r": (sum(t["r"] for t in closed) / len(closed)) if closed else 0.0,
        "pf": (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win else 0.0),
        "open": len(book.get("open", [])),
    }


def summary(book: dict) -> str:
    s = stats(book)
    if not s["trades"] and not s["open"]:
        return (
            "<b>عملکرد پیپر</b>\n"
            "───────────────\n"
            "هنوز معامله‌ای ثبت نشده.\n"
            "اولین سیگنال، اولین ردیف این دفتر می‌شود."
        )

    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    lines = [
        "<b>عملکرد پیپر</b>",
        "───────────────",
        f"<b>معاملات بسته‌شده:</b> {s['trades']}",
        f"<b>برد / باخت:</b> {s['wins']} / {s['losses']}"
        f" ({s['win_rate']:.1f}٪ برد)",
        f"<b>مجموع:</b> {s['sum_r']:+.2f}R · <b>میانگین:</b> {s['avg_r']:+.3f}R",
        f"<b>Profit Factor:</b> {pf}",
        f"<b>باز:</b> {s['open']} معامله",
    ]

    recent = book.get("closed", [])[-5:]
    if recent:
        lines.append("───────────────")
        lines.append("<b>۵ معامله آخر:</b> <i>به وقت تهران</i>")
        for trade in reversed(recent):
            mark = "🟢" if trade["r"] > 0 else "🔴"
            arrow = "▲" if trade["side"] > 0 else "▼"
            r_text = f"{trade['r']:+.2f}R".replace("-", "−")
            lines.append(
                f"{mark} {trade['symbol']} {arrow} {r_text}"
                f" · {when_tehran(trade.get('closed'))}"
            )

    open_trades = book.get("open", [])
    if open_trades:
        lines.append("───────────────")
        lines.append("<b>باز:</b>")
        for trade in open_trades[:5]:
            arrow = "▲" if trade["side"] > 0 else "▼"
            lines.append(
                f"• {trade['symbol']} {arrow} از {trade['entry']:,.6g}"
                f" · {when_tehran(trade.get('opened'))}"
            )
        if len(open_trades) > 5:
            lines.append(f"• و {len(open_trades) - 5} معامله دیگر")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    book = load()
    if "--stats" not in argv:
        closed = update(book)
        for trade in closed:
            print(f"closed {trade['symbol']:<12} {trade['reason']:<8} {trade['r']:+.2f}R")
        save(book)
    print(summary(book).replace("<b>", "").replace("</b>", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
