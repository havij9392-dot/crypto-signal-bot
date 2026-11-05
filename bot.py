"""
The interactive side of the bot.

The scanner only pushes signals. This module lets Amir talk back: a button asks
how much capital he is starting with, and from then on every signal is sized
from that number.

Commands
    /start          welcome, plus the capital keyboard
    /capital        ask again, with buttons
    /risk 0.5       risk per trade, in percent
    /equity 1250    update the live balance (peak is kept, so the brake works)
    /status         what the bot currently thinks
    /help           this list

The keyboard uses callback queries, so nothing is typed for the common case.

Runs for a fixed window then exits, because it is meant to be started by a
cron job rather than kept alive forever.

Usage:
    python bot.py            # poll for 240 seconds
    python bot.py 60         # shorter window
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request

import account
from notify import _credentials

API = "https://api.telegram.org/bot{token}/{method}"

WELCOME = (
    "سلام امیر ⚡\n\n"
    "برای شروع بگو با چه سرمایه‌ای می‌خواهی کار کنی.\n"
    "بعد از آن، هر سیگنال با حجم، مارجین و اهرم مخصوص همان سرمایه می‌آید."
)

CAPITAL_KEYBOARD = {
    "inline_keyboard": [
        [
            {"text": "۵۰۰ دلار", "callback_data": "cap:500"},
            {"text": "۱٬۰۰۰ دلار", "callback_data": "cap:1000"},
        ],
        [
            {"text": "۲٬۰۰۰ دلار", "callback_data": "cap:2000"},
            {"text": "۵٬۰۰۰ دلار", "callback_data": "cap:5000"},
        ],
        [{"text": "مقدار دیگر…", "callback_data": "cap:custom"}],
    ]
}

HELP = (
    "<b>دستورها</b>\n"
    "───────────────\n"
    "/capital — تعیین سرمایه (با دکمه)\n"
    "/risk 0.5 — ریسک هر معامله به درصد\n"
    "/equity 1250 — به‌روز کردن موجودی فعلی\n"
    "/status — وضعیت فعلی\n"
    "/help — همین لیست"
)


def _call(method: str, payload: dict) -> dict:
    token, _ = _credentials()
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API.format(token=token, method=method),
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"telegram HTTP {exc.code} on {method}: {body}")
        return {"ok": False}
    except Exception as exc:  # noqa: BLE001 - network hiccup
        print(f"{method} failed: {exc}")
        return {"ok": False}


def reply(chat_id: str, text: str, keyboard: dict | None = None) -> None:
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if keyboard:
        payload["reply_markup"] = keyboard
    _call("sendMessage", payload)


def parse_amount(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("٬", "").replace("$", "").replace("دلار", "").strip()
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    value = float(match.group())
    return value if value > 0 else None


def handle_callback(callback: dict, data: dict) -> None:
    chat_id = str(callback["message"]["chat"]["id"])
    callback_id = callback["id"]
    payload = callback.get("data", "")

    if payload.startswith("cap:"):
        choice = payload.split(":", 1)[1]
        if choice == "custom":
            data["awaiting"] = "capital"
            account.save(data)
            _call("answerCallbackQuery", {"callback_query_id": callback_id})
            reply(chat_id, "مقدار سرمایه را به دلار بنویس. مثال: <code>1500</code>")
            return

        amount = float(choice)
        account.set_capital(data, amount)
        data["awaiting"] = None
        account.save(data)
        _call("answerCallbackQuery", {"callback_query_id": callback_id, "text": "ثبت شد"})
        reply(chat_id, account.describe(data) + "\n\nهر سیگنال از این به بعد بر همین اساس سایز می‌شود.")
        return

    _call("answerCallbackQuery", {"callback_query_id": callback_id})


def handle_message(message: dict, data: dict) -> None:
    chat_id = str(message["chat"]["id"])
    text = (message.get("text") or "").strip()
    if not text:
        return

    if data.get("awaiting") == "capital":
        amount = parse_amount(text)
        if amount is None:
            reply(chat_id, "عدد را نفهمیدم. فقط رقم بفرست، مثلاً <code>1500</code>")
            return
        account.set_capital(data, amount)
        data["awaiting"] = None
        account.save(data)
        reply(chat_id, account.describe(data) + "\n\nهر سیگنال از این به بعد بر همین اساس سایز می‌شود.")
        return

    lowered = text.lower()

    if lowered.startswith("/start"):
        reply(chat_id, WELCOME, CAPITAL_KEYBOARD)
        return

    if lowered.startswith("/capital"):
        reply(chat_id, "سرمایه‌ات چقدر است؟", CAPITAL_KEYBOARD)
        return

    if lowered.startswith("/risk"):
        amount = parse_amount(text.split(maxsplit=1)[1]) if len(text.split()) > 1 else None
        if amount is None or amount > 10:
            reply(chat_id, "مثال: <code>/risk 0.5</code> — بین ۰٫۱ تا ۱۰ درصد")
            return
        data["risk_pct"] = round(amount, 2)
        account.save(data)
        reply(chat_id, account.describe(data))
        return

    if lowered.startswith("/equity"):
        amount = parse_amount(text.split(maxsplit=1)[1]) if len(text.split()) > 1 else None
        if amount is None:
            reply(chat_id, "مثال: <code>/equity 1250</code>")
            return
        account.set_equity(data, amount)
        account.save(data)
        reply(chat_id, account.describe(data))
        return

    if lowered.startswith("/status"):
        reply(chat_id, account.describe(data))
        return

    if lowered.startswith("/help"):
        reply(chat_id, HELP)
        return

    reply(chat_id, "دستور را نفهمیدم. /help را بزن.")


def poll(seconds: int) -> int:
    token, allowed_chat = _credentials()
    data = account.load()
    deadline = time.time() + seconds
    handled = 0

    print(f"polling for {seconds}s as chat {allowed_chat}")

    while time.time() < deadline:
        offset = int(data.get("last_update_id") or 0) + 1
        result = _call(
            "getUpdates",
            {"offset": offset, "timeout": 20, "allowed_updates": ["message", "callback_query"]},
        )
        if not result.get("ok"):
            time.sleep(2)
            continue

        for update in result.get("result", []):
            data["last_update_id"] = update["update_id"]

            chat_id = None
            if "message" in update:
                chat_id = str(update["message"]["chat"]["id"])
            elif "callback_query" in update:
                chat_id = str(update["callback_query"]["message"]["chat"]["id"])

            if chat_id != str(allowed_chat):
                print(f"ignoring update from chat {chat_id}")
                account.save(data)
                continue

            if "callback_query" in update:
                handle_callback(update["callback_query"], data)
            elif "message" in update:
                handle_message(update["message"], data)

            handled += 1
            account.save(data)

    account.save(data)
    print(f"handled {handled} update(s)")
    return 0


def main(argv: list[str]) -> int:
    seconds = int(argv[1]) if len(argv) > 1 else 240
    return poll(seconds)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
