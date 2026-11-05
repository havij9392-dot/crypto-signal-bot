"""Offline test of the bot handlers - no Telegram traffic, no state damage."""

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import account
import bot

BACKUP = os.path.join(HERE, "state", "account.json.testbak")
shutil.copy(account.ACCOUNT_FILE, BACKUP)

sent = []
bot._call = lambda method, payload: (sent.append((method, payload)), {"ok": True})[1]

CHAT = "6090974409"
data = {"capital": 0.0, "risk_pct": 1.0, "peak": 0.0, "last_update_id": 0, "awaiting": None}


def msg(text):
    sent.clear()
    bot.handle_message({"chat": {"id": CHAT}, "text": text}, data)
    return sent[-1][1].get("text", "") if sent else "<no reply>"


def cb(payload):
    sent.clear()
    bot.handle_callback(
        {"id": "1", "data": payload, "message": {"chat": {"id": CHAT}}}, data
    )
    return sent[-1][1].get("text", "") if sent else "<no reply>"


def show(label, text):
    print(f"\n=== {label} ===")
    print(text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""))


try:
    show("1. /start", msg("/start"))
    print("   keyboard rows:", len(sent[-1][1]["reply_markup"]["inline_keyboard"]))

    show("2. /capital -> press 2000", cb("cap:2000"))
    assert data["capital"] == 2000 and data["peak"] == 2000

    show("3. /equity -> type 1800", cb("eq:custom"))
    show("3b. type 1800", msg("1800"))
    assert data["capital"] == 1800 and data["peak"] == 2000, data
    assert "ترمز" not in msg("/status")

    show("4. press 'new balance' -> type 1400 (drawdown)", cb("eq:custom"))
    msg("1400")
    assert data["capital"] == 1400 and data["peak"] == 2000
    assert "ترمز" in msg("/status"), "brake should be active below 75% of peak"

    show("5. /capital custom -> 3000", cb("cap:custom"))
    show("5b. type 3000", msg("3000"))
    assert data["capital"] == 3000 and data["peak"] == 3000
    assert "ترمز" not in msg("/status")

    show("6. withdrawal 1000 via /capital", msg("/capital"))
    show("6b. type 1000", msg("1000"))
    assert data["capital"] == 1000 and data["peak"] == 1000
    assert "ترمز" not in msg("/status"), "a withdrawal must not trip the brake"

    show("7. /equity 1250 direct", msg("/equity 1250"))
    assert data["capital"] == 1250 and data["peak"] == 1250

    show("8. awaiting then /help clears it", msg("/equity"))
    msg("/help")
    assert data["awaiting"] is None

    show("9. /risk 0.5", msg("/risk 0.5"))
    assert data["risk_pct"] == 0.5

    show("10. eq:reset_peak", cb("eq:reset_peak"))
    assert data["peak"] == data["capital"]

    print("\nALL ASSERTIONS PASSED")
finally:
    shutil.copy(BACKUP, account.ACCOUNT_FILE)
    os.remove(BACKUP)
    print("state restored:", json.load(open(account.ACCOUNT_FILE, encoding="utf-8")))
