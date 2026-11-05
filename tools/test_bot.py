"""Offline tests: multi-user routing, admin panel, and the paper book.

No Telegram traffic and no state damage - users.json and paper.json are pointed
at temporary files for the duration of the run.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import account
import bot
import papertrade
import users

OWNER = "6090974409"
FRIEND = "777000111"
STRANGER = "555000222"

# ---------------------------------------------------------------- sandbox


def sandbox():
    tmpdir = tempfile.mkdtemp()
    users.STATE_DIR = tmpdir
    users.USERS_FILE = os.path.join(tmpdir, "users.json")
    papertrade.STATE_DIR = tmpdir
    papertrade.BOOK_FILE = os.path.join(tmpdir, "paper.json")


sandbox()
os.environ["TELEGRAM_CHAT_ID"] = OWNER

calls = []
LAST = {}
PER_CHAT = {}
BAR = {}


def fake_call(method, payload):
    calls.append((method, payload))
    if method in ("sendMessage", "editMessageText"):
        markup = payload.get("reply_markup")
        LAST["text"] = payload.get("text", "")
        LAST["keyboard"] = markup
        PER_CHAT[str(payload.get("chat_id"))] = (LAST["text"], markup)
        if isinstance(markup, dict) and "keyboard" in markup:
            BAR[str(payload.get("chat_id"))] = markup
    if method == "getMe":
        return {"ok": True, "result": {"username": "zakaisignal_bot"}}
    return {"ok": True, "result": {"message_id": 1}}


bot._call = fake_call

data = {
    "owner": OWNER,
    "bot_username": "zakaisignal_bot",
    "last_update_id": 0,
    "users": {},
    "invites": {},
}


def msg(chat_id, text, first_name="کاربر"):
    calls.clear()
    bot.handle_message(
        {"chat": {"id": chat_id, "first_name": first_name}, "text": text}, data
    )
    return LAST.get("text", "<no reply>")


def cb(chat_id, payload, message_id=10):
    calls.clear()
    bot.handle_callback(
        {
            "id": "cb1",
            "data": payload,
            "message": {"chat": {"id": chat_id}, "message_id": message_id},
        },
        data,
    )
    return LAST.get("text", "<no reply>")


def plain(text):
    for tag in ("<b>", "</b>", "<code>", "</code>"):
        text = text.replace(tag, "")
    return text


def show(label, text):
    print(f"\n=== {label} ===")
    print(plain(text))


def rows(keyboard):
    if isinstance(keyboard, str):
        keyboard = json.loads(keyboard)
    return [len(row) for row in keyboard["inline_keyboard"]]


# ------------------------------------------------------------------ tests

print("--- access control ---")
show("stranger without a code", msg(STRANGER, "/start"))
assert "خصوصی" in LAST["text"]

show("stranger with a bogus code", msg(STRANGER, "/start DEADBEEF"))
assert "خصوصی" in LAST["text"]
assert STRANGER not in data["users"]

# the real migration seeds the owner; do it by hand here
data["users"][OWNER] = users.new_user("امیر")
data["users"][OWNER]["capital"] = 1000.0
data["users"][OWNER]["peak"] = 1000.0

print("\n--- owner menu ---")
text = msg(OWNER, "/start")
show("owner /start (inline menu)", text)
assert "منوی اصلی" in LAST["text"] or "منوی کامل" in LAST["text"]
assert rows(LAST["keyboard"]) == [2, 2, 2, 1, 2], rows(LAST["keyboard"])

print("\n--- persistent bottom bar ---")
assert OWNER in BAR, "the menu must also be pinned to the bottom of the chat"
labels = [btn["text"] for row in BAR[OWNER]["keyboard"] for btn in row]
print("   bottom bar:", labels)
assert "☰ منو" in labels
assert BAR[OWNER].get("is_persistent") is True

text = msg(OWNER, "☰ منو")
assert "منوی اصلی" in text, "the ☰ منو button must reopen the menu"
assert rows(LAST["keyboard"]) == [2, 2, 2, 1, 2]

text = msg(OWNER, "📊 وضعیت")
assert "سرمایه فعلی" in text

text = msg(OWNER, "💰 سرمایه")
assert "💰 سرمایه" in text

text = msg(OWNER, "📉 عملکرد")
assert "عملکرد پیپر" in text

print("\n--- invite flow ---")
text = cb(OWNER, "adm:invite")
show("owner presses invite", text)
code = list(data["invites"])[0]
print("   code:", code)
assert data["invites"][code]["used_by"] is None

text = cb(FRIEND, "adm:invite")
assert "خصوصی" in text, "an unregistered chat gets nothing but the denial"
assert len(data["invites"]) == 1, "a non-admin must not mint invites"

text = msg(FRIEND, f"/start {code}", first_name="رضا")
welcome, welcome_kb = PER_CHAT[FRIEND]
show("friend redeems the code", welcome)
assert FRIEND in data["users"]
assert data["invites"][code]["used_by"] == FRIEND
assert FRIEND in BAR, "a new user must get the bottom bar"
assert "💰 سرمایه" in welcome, "a new user starts on the capital menu"
assert rows(welcome_kb) == [2, 2, 1, 1], rows(welcome_kb)
assert "کاربر جدید" in LAST["text"], "the owner must be told somebody joined"

text = msg(STRANGER, f"/start {code}", first_name="غریبه")
assert "خصوصی" in text, "an invite is single use"

print("\n--- per-user settings ---")
show("friend capital menu", cb(FRIEND, "m:cap"))
text = cb(FRIEND, "cap:2000")
show("friend presses 2000", text)
friend = data["users"][FRIEND]
assert friend["capital"] == 2000 and friend["peak"] == 2000
assert data["users"][OWNER]["capital"] == 1000, "one user's capital must not touch another's"

cb(FRIEND, "eq:custom")
show("friend types a new balance", msg(FRIEND, "1500"))
assert friend["capital"] == 1500 and friend["peak"] == 2000

show("friend picks risk 2%", cb(FRIEND, "risk:2"))
assert friend["risk_pct"] == 2.0

print("\n--- drawdown brake ---")
msg(FRIEND, "/equity 1400")
assert account.effective_risk(friend)[1] is True, "below 75% of peak, brake must fire"
print("   brake active, risk:", f"{account.effective_risk(friend)[0] * 100:.2f}%")
assert "ترمز" in cb(FRIEND, "m:status")

msg(FRIEND, "/capital 5000")
assert account.effective_risk(friend)[1] is False, "a deposit resets the peak, no brake"
print("   after /capital 5000, brake off")

print("\n--- signals toggle ---")
before = friend["signals"]
cb(FRIEND, "sig:toggle")
assert friend["signals"] is not before
print("   signals now:", friend["signals"])
cb(FRIEND, "sig:toggle")
assert friend["signals"] is before

print("\n--- admin panel ---")
text = cb(OWNER, "adm:users")
show("user list", text)
assert "کاربران (2)" in text

show("friend detail", cb(OWNER, f"adm:u:{FRIEND}"))
assert "5,000" in LAST["text"]

cb(FRIEND, f"adm:u:{OWNER}")
methods = [m for m, _ in calls]
assert methods == ["answerCallbackQuery"], f"non-admin got a panel: {methods}"

show("remove friend", cb(OWNER, f"adm:rm:{FRIEND}"))
assert FRIEND not in data["users"]

cb(OWNER, f"adm:rm:{OWNER}")
assert OWNER in data["users"], "the owner cannot remove himself"

print("\n--- paper book ---")


def row(ts, high, low, close, open_=None):
    return [
        int(ts.timestamp() * 1000),
        str(open_ or close),
        str(high),
        str(low),
        str(close),
        "1",
    ]


def trade(side, entry=100.0, stop_pct=5.0, opened="2026-11-05 00:00"):
    return {
        "id": "X",
        "symbol": "TESTUSDT",
        "side": side,
        "entry": entry,
        "stop_pct": stop_pct,
        "target_pct": stop_pct * 3,
        "opened": opened,
        "opened_at": opened,
    }


start = datetime(2026, 11, 5, 4, tzinfo=timezone.utc)
bar = timedelta(hours=4)

result = papertrade.walk(trade(1), [row(start, 116, 99, 115)])
assert result and result["reason"] == "target" and abs(result["r"] - 3.0) < 1e-9, result
print("   long target:", result["reason"], f"{result['r']:+.2f}R")

result = papertrade.walk(trade(1), [row(start, 101, 94, 94)])
assert result and result["reason"] == "stop" and abs(result["r"] + 1.0) < 1e-9, result
print("   long stop:", result["reason"], f"{result['r']:+.2f}R")

result = papertrade.walk(trade(-1), [row(start, 104, 84, 85)])
assert result and result["reason"] == "target" and abs(result["r"] - 3.0) < 1e-9, result
print("   short target:", result["reason"], f"{result['r']:+.2f}R")

result = papertrade.walk(trade(-1), [row(start, 106, 99, 106)])
assert result and result["reason"] == "stop" and abs(result["r"] + 1.0) < 1e-9, result
print("   short stop:", result["reason"], f"{result['r']:+.2f}R")

result = papertrade.walk(trade(1), [row(start, 116, 94, 100)])
assert result["reason"] == "stop", "when both are touched, assume the worst"
print("   ambiguous candle resolves to:", result["reason"])

bars = [row(start + bar * i, 101, 99, 100.5) for i in range(80)]
result = papertrade.walk(trade(1), bars)
assert result and result["reason"] == "timeout", result
print("   timeout after 60 bars:", f"{result['r']:+.2f}R")

same_bar = [row(datetime(2026, 11, 5, 0, tzinfo=timezone.utc), 999, 1, 100)]
assert papertrade.walk(trade(1), same_bar) is None, "the signal candle must be skipped"
print("   signal candle is skipped: ok")

print("\n--- book stats ---")
book = {
    "open": [],
    "closed": [
        {"symbol": "A", "side": 1, "r": 3.0, "reason": "target"},
        {"symbol": "B", "side": 1, "r": -1.0, "reason": "stop"},
        {"symbol": "C", "side": -1, "r": -1.0, "reason": "stop"},
        {"symbol": "D", "side": 1, "r": 3.0, "reason": "target"},
    ],
}
s = papertrade.stats(book)
assert s["trades"] == 4 and s["wins"] == 2 and s["losses"] == 2
assert abs(s["sum_r"] - 4.0) < 1e-9
assert abs(s["pf"] - 3.0) < 1e-9
print("   stats:", s)
print(plain(papertrade.summary(book)))

print("\nALL ASSERTIONS PASSED")
