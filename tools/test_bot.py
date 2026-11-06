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
from config import to_tehran

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

show("friend picks risk 2%", cb(FRIEND, "sz:risk:2"))
assert friend["risk_pct"] == 2.0 and friend["mode"] == "risk"

print("\n--- sizing modes ---")
# risk mode: the stop always costs the same dollars
small = account.sizing(1000, 0.01, 2.5)
wide = account.sizing(1000, 0.01, 8.5)
assert abs(small["risk_usd"] - 10) < 1e-9 and abs(wide["risk_usd"] - 10) < 1e-9
assert wide["margin"] < small["margin"], "a wider stop must buy a smaller position"
print(
    f"   risk 1%  · stop 2.5% -> margin ${small['margin']:.0f} · loss $10"
    f"   |   stop 8.5% -> margin ${wide['margin']:.0f} · loss $10"
)

# margin mode: the margin is fixed, so the loss floats
m50 = account.sizing(1000, 0.01, 8.5, 50.0)
m100 = account.sizing(1000, 0.01, 8.5, 100.0)
print(
    f"   margin $50  · stop 8.5% -> loss ${m50['risk_usd']:.0f}"
    f" ({m50['risk_pct']:.1f}% of account)"
)
print(
    f"   margin $100 · stop 8.5% -> loss ${m100['risk_usd']:.0f}"
    f" ({m100['risk_pct']:.1f}% of account)"
)
assert m50["risk_usd"] > wide["risk_usd"], "margin mode must risk more than 1%"
assert abs(m50["margin"] - 50) < 1e-6

# never risk more than the whole account, however wide the stop
wild = account.sizing(1000, 0.01, 40.0, 1000.0)
assert wild["risk_usd"] <= 1000 + 1e-6, wild

show("switch to fixed margin 50", cb(FRIEND, "sz:margin:50"))
assert friend["mode"] == "margin" and friend["margin"] == 50.0
sized = account.size_for(friend, 8.0)
print(
    f"   friend $5,000 · margin mode $50 · stop 8% -> loss ${sized['risk_usd']:.0f}"
    f" ({sized['risk_pct']:.1f}%)"
)
assert sized["mode"] == "margin"

show("switch back to risk 1%", cb(FRIEND, "sz:risk:1"))
assert friend["mode"] == "risk" and friend["risk_pct"] == 1.0

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
        {"symbol": "A", "side": 1, "r": 3.0, "reason": "target", "closed": "2026-11-05 00:00"},
        {"symbol": "B", "side": 1, "r": -1.0, "reason": "stop", "closed": "2026-11-05 04:00"},
        {"symbol": "C", "side": -1, "r": -1.0, "reason": "stop", "closed": "2026-11-05 08:00"},
        {"symbol": "D", "side": 1, "r": 3.0, "reason": "target", "closed": "2026-11-05 16:00"},
    ],
}
s = papertrade.stats(book)
assert s["trades"] == 4 and s["wins"] == 2 and s["losses"] == 2
assert abs(s["sum_r"] - 4.0) < 1e-9
assert abs(s["pf"] - 3.0) < 1e-9
print("   stats:", s)
summary_text = plain(papertrade.summary(book))
print(summary_text)
# 16:00 UTC candle closes at 20:00 UTC = 23:30 in Tehran
assert "23:30" in summary_text, "closed trades must be shown in Tehran time"
assert "19:30" not in summary_text, "a UTC stamp must never leak into the display"

print("\n--- timezone ---")
assert to_tehran("2026-11-05 16:00") == "2026-11-05 19:30"
assert to_tehran("2026-11-05 16:00", 4) == "2026-11-05 23:30"
assert to_tehran("2026-11-05T20:33:20Z") == "2026-11-06 00:03", "ISO stamps roll over midnight"
assert to_tehran("") == "" and to_tehran("nonsense") == "nonsense"
print("   candle 16:00 UTC closes", to_tehran("2026-11-05 16:00", 4), "Tehran")
print("   joined stamp", to_tehran("2026-11-05T20:33:20Z"), "Tehran")

print("\n--- late signals and heartbeat ---")
import pandas as pd

import scanner

frame = pd.DataFrame(
    {
        "open_time": ["2026-11-05 08:00", "2026-11-05 12:00", "2026-11-05 16:00"],
        "close": [100.0, 101.0, 102.0],
        "atr": [3.0, 3.0, 3.0],
        "rsi": [50.0, 55.0, 60.0],
    }
)

fresh = scanner.build_plan("XUSDT", frame, 1, -1)
late = scanner.build_plan("XUSDT", frame, 1, -2)
print("   fresh signal labels candle:", fresh["candle"], "| late:", fresh["late"])
print("   late signal labels candle:", late["candle"], "| late:", late["late"])
assert fresh["candle"] == "2026-11-05 16:00" and fresh["late"] is False
assert late["candle"] == "2026-11-05 12:00" and late["late"] is True
assert late["price"] == 102.0, "a late signal must still quote the price you can get now"
assert late["rsi"] == 55.0, "but its indicator values belong to the signal candle"

beat = os.path.join(tempfile.mkdtemp(), "heartbeat.json")
bot.HEARTBEAT = beat

print("   no heartbeat yet:", bot.last_scan_line())
assert "نامشخص" in bot.last_scan_line()

now = datetime.now(timezone.utc)
with open(beat, "w", encoding="utf-8") as handle:
    json.dump({"last_scan": now.strftime("%Y-%m-%dT%H:%M:%SZ")}, handle)
print("   fresh heartbeat:", bot.last_scan_line())
assert "دقیقه پیش" in bot.last_scan_line()

old = now - timedelta(hours=5)
with open(beat, "w", encoding="utf-8") as handle:
    json.dump({"last_scan": old.strftime("%Y-%m-%dT%H:%M:%SZ")}, handle)
print("   stale heartbeat:", bot.last_scan_line())
assert "عقب است" in bot.last_scan_line()

print("\n--- result replies ---")
sent = []


def fake_send(text, chat_id=None, silent=False, keyboard=None, reply_to=None):
    sent.append({"chat": chat_id, "text": text, "reply_to": reply_to})
    return 1000 + len(sent)  # pretend Telegram handed back a message id


scanner.send = fake_send

anchors = {
    OWNER: {"message_id": 501, "risk_usd": 98.0},
    FRIEND: {"message_id": 777, "risk_usd": 50.0},
}
plan = {
    "symbol": "SOLUSDT",
    "side": 1,
    "price": 100.0,
    "stop_pct": 4.0,
    "target_pct": 12.0,
    "candle": "2026-11-05 16:00",
    "rsi": 55.0,
}
fresh_book = {"open": [], "closed": []}
trade = papertrade.record(fresh_book, plan, anchors)
assert trade and trade["recipients"] == anchors, "the anchor must survive into the book"
assert papertrade.record(fresh_book, plan, anchors) is None, "one open trade per symbol"

closed_trade = dict(trade)
closed_trade.update({"r": 3.0, "reason": "target", "closed": "2026-11-05 20:00", "exit": 112.0})

reply_count = scanner.announce_close(closed_trade, [(OWNER, {}), (FRIEND, {})], do_send=True)
assert reply_count == 2, reply_count
assert [m["reply_to"] for m in sent] == [501, 777], "each user gets a reply under their own message"
assert sent[0]["chat"] == OWNER and sent[1]["chat"] == FRIEND
assert "+$294" in sent[0]["text"], "3R against $98 risk is +$294"
assert "+$150" in sent[1]["text"], "3R against $50 risk is +$150"
print("   replies anchored to:", [m["reply_to"] for m in sent])
print("   owner text has:", [line for line in sent[0]["text"].splitlines() if "سود/زیان" in line])

sent.clear()
loser = dict(closed_trade, r=-1.0, reason="stop")
scanner.announce_close(loser, [(OWNER, {})], do_send=True)
assert "−$98" in sent[0]["text"], "a loss must print with a minus, not a dash"
print("   loss line:", [line for line in sent[0]["text"].splitlines() if "سود/زیان" in line])

sent.clear()
old_trade = {"symbol": "OLDUSDT", "side": -1, "r": 1.0, "reason": "target",
             "closed": "2026-11-05 20:00"}
scanner.announce_close(old_trade, [(OWNER, {})], do_send=True)
assert sent[0]["reply_to"] is None, "a trade with no anchor falls back to a broadcast"
print("   trade booked before anchors existed: broadcast, not a reply")

sent.clear()
scanner.announce_close(closed_trade, [], do_send=False)
assert not sent, "a dry run must not send anything"

print("\n--- entry committee ---")
import numpy as np
import pandas as pd

import committee
from features import add_all


def trend_frame(step=1.003, amp_scale=1.0, volume=1000.0):
    """A synthetic uptrend whose volatility breathes, so ATR rank is meaningful."""
    n = 400
    k = np.arange(n)
    amp = (1.0 + 0.9 * np.sin(k / 9.0)) * amp_scale
    close = 100 * np.power(step, k) + amp * np.where(k % 2 == 0, 0.5, -0.5)
    return add_all(pd.DataFrame({
        "open": close - 0.1, "high": close + amp, "low": close - amp,
        "close": close, "volume": np.full(n, volume),
    }))


good = trend_frame()
decision = committee.judge(good, 399, 1)
assert decision["pass"], decision["votes"]
assert decision["approved"] == 3, "a clean trend should carry every vote"
print("   clean trend:", decision["approved"], "of", len(decision["votes"]), "approve")

flat = trend_frame(step=1.0)
blocked = committee.judge(flat, 399, 1)
assert not blocked["pass"], "a flat market must be blocked"
assert any(b["key"] == "adx" for b in blocked["blockers"]), blocked["blockers"]
print("   flat market blocked by:", [b["name"] for b in blocked["blockers"]])

quiet = trend_frame(volume=200.0)          # far below its own 20-bar average
quiet["volume"] = 200.0
quiet["vol_sma20"] = 1000.0
blocked = committee.judge(quiet, 399, 1)
assert any(b["key"] == "volume" for b in blocked["blockers"]), blocked["blockers"]
print("   thin volume blocked by:", [b["name"] for b in blocked["blockers"]])

wide = committee.add_votes(
    decision,
    [committee.breadth(0.20, 1, committee.PARAMS), committee.liquidity(50e6, 3e6)],
)
assert not wide["pass"], "a market where only 20% agree must not pass"
assert wide["blockers"][0]["key"] == "breadth"
print("   narrow breadth blocked by:", [b["name"] for b in wide["blockers"]])

# the exposure guard: the sixth copy of the same bet
open_book = [{"side": 1}] * 5
assert committee.exposure(open_book, 1)["vote"] == committee.VETO
assert committee.exposure(open_book, -1)["vote"] == committee.APPROVE
print("   five longs open: a sixth long is refused, a short is allowed")

rendered = scanner.format_message(
    {
        "symbol": "SOLUSDT", "side": 1, "price": 100.0, "stop_pct": 4.0,
        "target_pct": 12.0, "rsi": 60.0, "candle": "2026-11-05 16:00",
        "late": False, "max_lev": 10, "loss_at_max_lev": 40.0,
        "volume_usd": 48e6, "decision": decision,
    },
    {"capital": 1000.0, "risk_pct": 1.0, "peak": 1000.0, "mode": "risk", "margin": 50.0},
)
assert "هیئت ورود" in rendered, "the message must show the committee"
assert "قدرت روند" in rendered and "گستردگی" not in rendered
print("   signal message carries the committee block")

print("\nALL ASSERTIONS PASSED")
