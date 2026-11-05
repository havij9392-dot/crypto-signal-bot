"""
Interactive side of the bot.

Everything is a button - nothing has to be typed. Menus are edited in place, so
the chat does not fill up with dead keyboards.

The bot serves a small private group. The owner (TELEGRAM_CHAT_ID) opens the
admin panel, mints an invite link, and whoever opens that link gets their own
capital, risk and signal settings. Signals are broadcast to everyone, sized from
each person's own capital.

Commands still work as shortcuts, but they are optional:

    /start [code]   open the menu, or redeem an invite code
    /menu           same as /start without a code
    /capital 1000   shortcut for the capital button
    /equity 1250    shortcut for the balance button
    /risk 1         shortcut for the risk button
    /stats          paper-trading record
    /help           what everything does

Runs for a fixed window then exits, because it is started by a cron job rather
than kept alive forever.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import account
import papertrade
import users
from notify import _credentials

API = "https://api.telegram.org/bot{token}/{method}"

HERE = os.path.dirname(os.path.abspath(__file__))
HEARTBEAT = os.path.join(HERE, "state", "heartbeat.json")


# ---------------------------------------------------------------- transport


def _call(method: str, payload: dict, retries: int = 3) -> dict:
    token, _ = _credentials()
    data = json.dumps(payload).encode("utf-8")
    last: dict = {"ok": False}

    for attempt in range(retries):
        request = urllib.request.Request(
            API.format(token=token, method=method),
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"telegram HTTP {exc.code} on {method}: {detail}")
            last = {"ok": False, "description": detail}
            # 429 is a burst limit: several buttons pressed at once. Wait it out.
            if exc.code == 429:
                try:
                    wait = json.loads(detail).get("parameters", {}).get("retry_after", 2)
                except json.JSONDecodeError:
                    wait = 2
                time.sleep(min(10, int(wait) + 1))
                continue
            if exc.code in (400, 403):
                return last  # retrying will not help
        except Exception as exc:  # noqa: BLE001 - network hiccup
            print(f"{method} failed: {exc}")
            last = {"ok": False, "description": str(exc)}
            time.sleep(1.5 * (attempt + 1))

    return last


def send_text(
    chat_id: str,
    text: str,
    keyboard: dict | None = None,
    reply: dict | None = None,
) -> int | None:
    payload = {
        "chat_id": str(chat_id),
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if keyboard:
        payload["reply_markup"] = keyboard
    elif reply:
        payload["reply_markup"] = reply
    result = _call("sendMessage", payload)
    if result.get("ok"):
        return result["result"]["message_id"]
    return None


def show(
    chat_id: str, text: str, keyboard: dict | None = None, message_id: int | None = None
) -> None:
    """Edit the menu in place when possible, otherwise send a fresh one."""
    if message_id:
        payload = {
            "chat_id": str(chat_id),
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if keyboard:
            payload["reply_markup"] = keyboard
        result = _call("editMessageText", payload)
        if result.get("ok"):
            return
        if "message is not modified" in (result.get("description") or ""):
            return
    send_text(chat_id, text, keyboard)


# -------------------------------------------------------------------- menus


def b(text: str, data: str) -> dict:
    return {"text": text, "callback_data": data}


def kb_main(user: dict, admin: bool) -> dict:
    on = user.get("signals", True)
    rows = [
        [b("💰 سرمایه", "m:cap"), b("📊 وضعیت", "m:status")],
        [b("🎚 اندازه پوزیشن", "m:size"), b("📈 موجودی", "m:eq")],
        [b("📉 عملکرد", "m:perf"), b("❓ راهنما", "m:help")],
        [b(f"🔔 سیگنال‌ها: {'روشن' if on else 'خاموش'}", "sig:toggle")],
    ]
    if admin:
        rows.append([b("👥 کاربران", "adm:users"), b("🔗 دعوت", "adm:invite")])
    return {"inline_keyboard": rows}


def kb_capital() -> dict:
    return {
        "inline_keyboard": [
            [b("۵۰۰ دلار", "cap:500"), b("۱٬۰۰۰ دلار", "cap:1000")],
            [b("۲٬۰۰۰ دلار", "cap:2000"), b("۵٬۰۰۰ دلار", "cap:5000")],
            [b("✏️ مقدار دیگر", "cap:custom")],
            [b("⬅️ برگشت", "m:home")],
        ]
    }


def kb_risk() -> dict:
    return {
        "inline_keyboard": [
            [
                b("۰٫۵٪", "risk:0.5"),
                b("۱٪", "risk:1"),
                b("۱٫۵٪", "risk:1.5"),
                b("۲٪", "risk:2"),
            ],
            [b("✏️ مقدار دیگر", "risk:custom")],
            [b("⬅️ برگشت", "m:home")],
        ]
    }


def kb_size(user: dict) -> dict:
    mode = user.get("mode") or "risk"
    risk = float(user.get("risk_pct") or 1.0)
    margin = float(user.get("margin") or 50.0)

    def mark(label: str, active: bool) -> str:
        return f"✅ {label}" if active else label

    return {
        "inline_keyboard": [
            [
                b(mark("ریسک ۱٪", mode == "risk" and risk == 1.0), "sz:risk:1"),
                b(mark("ریسک ۲٪", mode == "risk" and risk == 2.0), "sz:risk:2"),
            ],
            [
                b(mark("مارجین ۵۰$", mode == "margin" and margin == 50.0), "sz:margin:50"),
                b(mark("مارجین ۱۰۰$", mode == "margin" and margin == 100.0), "sz:margin:100"),
            ],
            [b("✏️ ریسک دلخواه", "sz:risk:custom"), b("✏️ مارجین دلخواه", "sz:margin:custom")],
            [b("⬅️ برگشت", "m:home")],
        ]
    }


def render_size(user: dict) -> str:
    capital = float(user.get("capital") or 0)
    mode = user.get("mode") or "risk"
    risk = float(user.get("risk_pct") or 1.0)
    margin = float(user.get("margin") or 50.0)

    if mode == "margin":
        now = f"مارجین ثابت ${margin:,.0f}"
    else:
        now = f"ریسک ثابت {risk:.2f}٪"

    body = (
        "<b>🎚 اندازه پوزیشن</b>\n"
        "───────────────\n"
        f"<b>الان:</b> {now}\n\n"
        "<b>ریسک ثابت</b> — در هر معامله همان مقدار دلار را خطر می‌کنی. حد ضررِ دورتر "
        "پوزیشن کوچک‌تری می‌خرد. این همان حالتی است که ۵ ماه تست با آن اندازه‌گیری شد.\n\n"
        "<b>مارجین ثابت</b> — خودت دلار مارجین را می‌گویی، پس پوزیشن بزرگ‌تر می‌شود. "
        "اما ضرر هم بزرگ می‌شود و با فاصله حد ضرر عوض می‌شود.\n"
    )

    if capital > 0:
        body += "\n<b>با سرمایه فعلی و حد ضرر ۸٪:</b>\n"
        sample = account.sizing(capital, risk / 100.0, 8.0)
        body += (
            f"• ریسک {risk:.2f}٪ → مارجین ${sample['margin']:,.0f} · "
            f"ضرر ${sample['risk_usd']:,.0f}\n"
        )
        big = account.sizing(capital, risk, 8.0, 50.0)
        body += (
            f"• مارجین $50 → ضرر ${big['risk_usd']:,.0f} "
            f"({big['risk_pct']:.1f}٪ حساب)\n"
        )
        big100 = account.sizing(capital, risk, 8.0, 100.0)
        body += (
            f"• مارجین $100 → ضرر ${big100['risk_usd']:,.0f} "
            f"({big100['risk_pct']:.1f}٪ حساب)\n"
        )
        body += "\n<i>در تست ۵ ماهه، ریسک ۴٪ افت ۹۱٪ داد و ریسک ۸٪ حساب را نابود کرد.</i>"
    return body


def kb_equity() -> dict:
    return {
        "inline_keyboard": [
            [b("✏️ موجودی جدید", "eq:custom")],
            [b("♻️ بازنشانی اوج", "eq:reset_peak")],
            [b("⬅️ برگشت", "m:home")],
        ]
    }


def kb_back(data: str = "m:home") -> dict:
    return {"inline_keyboard": [[b("⬅️ برگشت", data)]]}


# The always-visible bar at the bottom of the chat. Inline menus scroll away;
# this one does not, so the menu is never lost.
REPLY_MENU = {
    "keyboard": [
        [{"text": "☰ منو"}, {"text": "📊 وضعیت"}],
        [{"text": "💰 سرمایه"}, {"text": "📉 عملکرد"}],
    ],
    "resize_keyboard": True,
    "is_persistent": True,
    "input_field_placeholder": "یکی از دکمه‌های پایین را بزن",
}

MENU_TEXTS = {"☰ منو", "منو", "💰 سرمایه", "📊 وضعیت", "📉 عملکرد"}

BAR_HINT = "\n\n⬇️ دکمه‌های پایین صفحه همیشه در دسترس‌اند؛ «☰ منو» را بزن."


def kb_users(data: dict) -> dict:
    rows = []
    for chat_id, user in data["users"].items():
        tag = " 👑" if chat_id == str(data.get("owner")) else ""
        capital = float(user.get("capital") or 0)
        rows.append([b(f"{user.get('name', '?')}{tag} — ${capital:,.0f}", f"adm:u:{chat_id}")])
    rows.append([b("🔗 دعوت", "adm:invite"), b("⬅️ برگشت", "m:home")])
    return {"inline_keyboard": rows}


# --------------------------------------------------------------------- text


HOME_TEXT = (
    "<b>منوی اصلی</b>\n"
    "───────────────\n"
    "هر کاری لازم داری از همین دکمه‌ها انجام بده.\n"
    "هیچ دستوری لازم نیست تایپ کنی."
)

HELP_TEXT = (
    "<b>راهنما</b>\n"
    "───────────────\n"
    "<b>💰 سرمایه</b> — وقتی پول واریز یا برداشت می‌کنی. اوج حساب هم روی عدد جدید می‌رود، "
    "پس برداشت به‌عنوان ضرر حساب نمی‌شود.\n\n"
    "<b>📈 موجودی</b> — وقتی موجودی با سود یا زیان معاملات عوض شده. اوج حساب دست‌نخورده "
    "می‌ماند تا ترمز افت افت واقعی را ببیند.\n\n"
    "<b>🎚 اندازه پوزیشن</b> — دو حالت دارد:\n"
    "«ریسک ثابت» یعنی در هر معامله همان مقدار دلار را خطر می‌کنی؛ حد ضرر دورتر، "
    "پوزیشن کوچک‌تر می‌خرد.\n"
    "«مارجین ثابت» یعنی خودت دلار مارجین را می‌گویی، پس پوزیشن بزرگ‌تر و ضرر هم "
    "بزرگ‌تر می‌شود.\n\n"
    "<b>📉 عملکرد</b> — کارنامه واقعی استراتژی روی سیگنال‌های ارسال‌شده.\n\n"
    "<b>🔔 سیگنال‌ها</b> — روشن یا خاموش کردن دریافت سیگنال.\n\n"
    "<b>اهرم</b> — از فاصله حد ضرر حساب می‌شود و هرگز از ۱۰× بالاتر نمی‌رود، "
    "پس حد ضرر همیشه قبل از لیکویید می‌خورد."
)

DENIED = (
    "این ربات خصوصی است.\n"
    "برای دسترسی، از سازنده‌اش لینک دعوت بگیر و روی همان لینک بزن."
)

WELCOME = (
    "سلام {name} ⚡\n"
    "───────────────\n"
    "من سیگنال‌های فیوچرز را روی تایم‌فریم ۴ ساعته می‌فرستم و برای هر سیگنال، "
    "حجم، اهرم و مارجین مخصوص سرمایه خودت را حساب می‌کنم.\n\n"
    "اول سرمایه‌ات را بگو، بعد بقیه‌اش با دکمه."
)


def money(value: float) -> str:
    return f"${value:,.0f}"


# ---------------------------------------------------------------- utilities


PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def parse_amount(text: str) -> float | None:
    cleaned = (
        text.translate(PERSIAN_DIGITS)
        .replace("٫", ".")
        .replace(",", "")
        .replace("٬", "")
        .replace("$", "")
        .replace("٪", "")
        .replace("%", "")
        .replace("دلار", "")
        .strip()
    )
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        value = float(match.group())
    except ValueError:
        return None
    return value if value > 0 else None


def arg_after(text: str, command: str) -> str:
    """Whatever follows the command, with or without a space: '/equity200' works."""
    return text[len(command):].strip()


def remember_bot_username(data: dict) -> None:
    if data.get("bot_username"):
        return
    result = _call("getMe", {})
    if result.get("ok"):
        data["bot_username"] = result["result"].get("username", "")
        users.save(data)


def notify_owner(data: dict, text: str) -> None:
    owner = data.get("owner")
    if owner:
        send_text(owner, text)


# ---------------------------------------------------------------- rendering


def last_scan_line() -> str:
    """How long ago the scanner last ran - so a dead scanner is visible in chat."""
    try:
        with open(HEARTBEAT, encoding="utf-8") as handle:
            beat = json.load(handle)
        when = datetime.strptime(beat["last_scan"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (OSError, KeyError, ValueError):
        return "📡 آخرین اسکن: نامشخص"

    minutes = int((datetime.now(timezone.utc) - when).total_seconds() // 60)
    if minutes < 0:
        minutes = 0
    if minutes < 90:
        return f"📡 آخرین اسکن: {minutes} دقیقه پیش"
    return f"⚠️ آخرین اسکن: {minutes // 60} ساعت پیش — اسکن عقب است"


def render_home(user: dict, admin: bool) -> str:
    capital = float(user.get("capital") or 0)
    risk_pct, braked = account.effective_risk(user)
    line = f"<b>{user.get('name', 'کاربر')}</b>"
    if capital > 0:
        line += (
            f" · {money(capital)} · ریسک {risk_pct * 100:.2f}٪"
            + (" 🔻ترمز افت" if braked else "")
        )
    else:
        line += " · سرمایه ثبت نشده"
    return f"{HOME_TEXT}\n───────────────\n{line}\n{last_scan_line()}"


def render_capital(user: dict) -> str:
    capital = float(user.get("capital") or 0)
    current = money(capital) if capital > 0 else "ثبت نشده"
    return (
        "<b>💰 سرمایه</b>\n"
        "───────────────\n"
        f"<b>فعلی:</b> {current}\n\n"
        "این را وقتی عوض کن که پول <b>واریز یا برداشت</b> می‌کنی.\n"
        "اوج حساب هم روی عدد جدید می‌رود، پس برداشت ضرر حساب نمی‌شود.\n\n"
        "یک دکمه بزن یا عدد را تایپ کن."
    )


def render_risk(user: dict) -> str:
    risk_pct, braked = account.effective_risk(user)
    base = float(user.get("risk_pct") or 1.0)
    capital = float(user.get("capital") or 0)
    body = "<b>⚙️ ریسک هر معامله</b>\n───────────────\n" f"<b>تنظیم‌شده:</b> {base:.2f}٪\n"
    if braked:
        body += f"<b>الان:</b> {risk_pct * 100:.2f}٪ (ترمز افت فعال)\n"
    if capital > 0:
        body += f"<b>یعنی:</b> {money(capital * risk_pct)} در هر معامله\n"
    body += "\n۱٪ نقطه شروع خوبی است. بالای ۲٪ با چند باخت پشت‌سرهم حساب را می‌برد."
    return body


def render_equity(user: dict) -> str:
    capital = float(user.get("capital") or 0)
    peak = float(user.get("peak") or 0)
    body = "<b>📈 موجودی</b>\n───────────────\n"
    if capital > 0:
        body += f"<b>فعلی:</b> {money(capital)} · <b>اوج:</b> {money(peak)}\n"
        if peak > capital:
            body += f"<b>افت از اوج:</b> {(capital / peak - 1) * 100:.1f}٪\n"
        body += "\n"
    body += (
        "این را وقتی بزن که موجودی‌ات با <b>سود یا زیان معاملات</b> عوض شده.\n"
        "اوج حساب دست‌نخورده می‌ماند، تا ترمز افت افت واقعی را ببیند."
    )
    return body


def render_performance() -> str:
    return papertrade.summary(papertrade.load())


def render_user_detail(data: dict, chat_id: str) -> str:
    user = data["users"].get(chat_id)
    if not user:
        return "این کاربر پیدا نشد."
    risk_pct, braked = account.effective_risk(user)
    lines = [
        f"<b>👤 {user.get('name', '?')}</b>",
        "───────────────",
        f"<b>شناسه:</b> <code>{chat_id}</code>",
        f"<b>سرمایه:</b> {money(float(user.get('capital') or 0))}",
        f"<b>ریسک:</b> {risk_pct * 100:.2f}٪" + (" 🔻ترمز افت" if braked else ""),
        f"<b>سیگنال:</b> {'روشن' if user.get('signals', True) else 'خاموش'}",
        f"<b>عضویت:</b> {user.get('joined', '?')}",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------- handlers


def apply_capital(data: dict, chat_id: str, amount: float, message_id: int | None) -> None:
    user = data["users"][chat_id]
    account.set_capital(user, amount)
    user["awaiting"] = None
    users.save(data)
    show(
        chat_id,
        f"✅ سرمایه روی {money(amount)} تنظیم شد.\n"
        "اوج حساب هم همین عدد شد، پس ترمز افت از اینجا حساب می‌شود.\n\n"
        + account.describe(user),
        kb_back(),
        message_id,
    )


def apply_equity(data: dict, chat_id: str, amount: float, message_id: int | None) -> None:
    user = data["users"][chat_id]
    account.set_equity(user, amount)
    user["awaiting"] = None
    users.save(data)
    show(
        chat_id,
        f"✅ موجودی روی {money(amount)} به‌روز شد.\n"
        "اوج حساب دست‌نخورده ماند.\n\n" + account.describe(user),
        kb_back(),
        message_id,
    )


def apply_risk(data: dict, chat_id: str, amount: float, message_id: int | None) -> None:
    user = data["users"][chat_id]
    user["risk_pct"] = round(min(account.MAX_RISK_PCT, max(account.MIN_RISK_PCT, amount)), 2)
    user["mode"] = "risk"
    user["awaiting"] = None
    users.save(data)
    show(chat_id, "✅ ذخیره شد.\n\n" + render_size(user), kb_size(user), message_id)


def apply_margin(data: dict, chat_id: str, amount: float, message_id: int | None) -> None:
    user = data["users"][chat_id]
    user["mode"] = "margin"
    user["margin"] = round(min(account.MAX_MARGIN, max(1.0, amount)), 2)
    user["awaiting"] = None
    users.save(data)
    show(chat_id, "✅ ذخیره شد.\n\n" + render_size(user), kb_size(user), message_id)


def handle_callback(callback: dict, data: dict) -> None:
    message = callback.get("message")
    if not message:
        _call("answerCallbackQuery", {"callback_query_id": callback["id"]})
        return

    chat_id = str(message["chat"]["id"])
    message_id = message["message_id"]
    payload = callback.get("data", "")

    # A button under a signal message must open a fresh menu: editing it would
    # overwrite the signal itself, and that message is the user's record.
    if "سیگنال" in (message.get("text") or ""):
        message_id = None

    _call("answerCallbackQuery", {"callback_query_id": callback["id"]})

    user = users.get(data, chat_id)
    if not user:
        send_text(chat_id, DENIED)
        return

    admin = users.is_admin(data, chat_id)

    if payload == "m:home":
        show(chat_id, render_home(user, admin), kb_main(user, admin), message_id)
        return

    if payload == "m:cap":
        show(chat_id, render_capital(user), kb_capital(), message_id)
        return

    if payload == "m:risk":
        show(chat_id, render_risk(user), kb_risk(), message_id)
        return

    if payload == "m:size":
        show(chat_id, render_size(user), kb_size(user), message_id)
        return

    if payload.startswith("sz:"):
        _, mode, value = payload.split(":", 2)
        if mode == "risk":
            if value == "custom":
                user["awaiting"] = "risk"
                users.save(data)
                show(
                    chat_id,
                    "درصد ریسک را بنویس.\nمثال: <code>1.5</code>",
                    kb_back("m:size"),
                    message_id,
                )
                return
            apply_risk(data, chat_id, float(value), message_id)
            return
        if value == "custom":
            user["awaiting"] = "margin"
            users.save(data)
            show(
                chat_id,
                "دلار مارجین هر معامله را بنویس.\nمثال: <code>75</code>",
                kb_back("m:size"),
                message_id,
            )
            return
        apply_margin(data, chat_id, float(value), message_id)
        return

    if payload == "m:eq":
        show(chat_id, render_equity(user), kb_equity(), message_id)
        return

    if payload == "m:status":
        show(chat_id, account.describe(user), kb_back(), message_id)
        return

    if payload == "m:help":
        show(chat_id, HELP_TEXT, kb_back(), message_id)
        return

    if payload == "m:perf":
        show(chat_id, render_performance(), kb_back(), message_id)
        return

    if payload == "sig:toggle":
        user["signals"] = not user.get("signals", True)
        users.save(data)
        note = "🔔 سیگنال‌ها روشن شد." if user["signals"] else "🔕 سیگنال‌ها خاموش شد."
        show(
            chat_id,
            render_home(user, admin) + "\n\n" + note,
            kb_main(user, admin),
            message_id,
        )
        return

    if payload.startswith("cap:"):
        choice = payload.split(":", 1)[1]
        if choice == "custom":
            user["awaiting"] = "capital"
            users.save(data)
            show(
                chat_id,
                "مقدار سرمایه را به دلار بنویس.\nمثال: <code>1500</code>",
                kb_back(),
                message_id,
            )
            return
        apply_capital(data, chat_id, float(choice), message_id)
        return

    if payload.startswith("risk:"):
        choice = payload.split(":", 1)[1]
        if choice == "custom":
            user["awaiting"] = "risk"
            users.save(data)
            show(
                chat_id,
                "درصد ریسک را بنویس.\nمثال: <code>0.75</code>",
                kb_back(),
                message_id,
            )
            return
        apply_risk(data, chat_id, float(choice), message_id)
        return

    if payload.startswith("eq:"):
        choice = payload.split(":", 1)[1]
        if choice == "custom":
            user["awaiting"] = "equity"
            users.save(data)
            show(
                chat_id,
                "موجودی فعلی را بنویس.\nمثال: <code>1250</code>",
                kb_back(),
                message_id,
            )
            return
        account.reset_peak(user)
        user["awaiting"] = None
        users.save(data)
        show(
            chat_id,
            "✅ اوج حساب روی موجودی فعلی تنظیم شد.\n\n" + account.describe(user),
            kb_back(),
            message_id,
        )
        return

    # ------------------------------------------------------------ admin only
    if not admin:
        return

    if payload == "adm:users":
        if not data["users"]:
            show(chat_id, "هنوز کاربری ثبت نشده.", kb_users(data), message_id)
            return
        show(
            chat_id,
            f"<b>👥 کاربران ({len(data['users'])})</b>\n"
            "───────────────\n"
            "روی هر کاربر بزن تا جزئیاتش را ببینی.",
            kb_users(data),
            message_id,
        )
        return

    if payload.startswith("adm:u:"):
        target = payload.split(":", 2)[2]
        text = render_user_detail(data, target)
        if target == str(data.get("owner")):
            keyboard = {"inline_keyboard": [[b("⬅️ برگشت", "adm:users")]]}
        else:
            keyboard = {
                "inline_keyboard": [
                    [b("🗑 حذف دسترسی", f"adm:rm:{target}")],
                    [b("⬅️ برگشت", "adm:users")],
                ]
            }
        show(chat_id, text, keyboard, message_id)
        return

    if payload.startswith("adm:rm:"):
        target = payload.split(":", 2)[2]
        name = (data["users"].get(target) or {}).get("name", target)
        if users.remove_user(data, target):
            users.save(data)
            send_text(target, "دسترسی شما به این ربات بسته شد.")
            show(chat_id, f"🗑 {name} حذف شد.", kb_users(data), message_id)
        else:
            show(chat_id, "این کاربر را نمی‌شود حذف کرد.", kb_users(data), message_id)
        return

    if payload == "adm:invite":
        code = users.make_invite(data, chat_id)
        users.save(data)
        link = users.invite_link(data, code)
        show(
            chat_id,
            "<b>🔗 لینک دعوت</b>\n"
            "───────────────\n"
            f"<code>{link}</code>\n\n"
            f"<b>کد:</b> <code>{code}</code>\n\n"
            "این لینک را برای هر کسی می‌خواهی بفرست. یک‌بار مصرف است و بعد از ۷۲ ساعت منقضی می‌شود.\n"
            "هر کسی بازش کند، حساب خودش را می‌گیرد و سیگنال‌ها را با سرمایه خودش دریافت می‌کند.",
            {
                "inline_keyboard": [
                    [b("♻️ لینک جدید", "adm:invite")],
                    [b("⬅️ برگشت", "m:home")],
                ]
            },
            message_id,
        )
        return


def handle_message(message: dict, data: dict) -> None:
    chat_id = str(message["chat"]["id"])
    text = (message.get("text") or "").strip()
    if not text:
        return

    lowered = text.lower()
    parts = text.split(maxsplit=1)

    user = users.get(data, chat_id)

    # ------------------------------------------------------- not registered
    if not user:
        if lowered.startswith("/start"):
            code = parts[1].strip() if len(parts) > 1 else ""
            name = (
                message["chat"].get("first_name")
                or message["chat"].get("username")
                or "کاربر"
            )
            if code and users.redeem(data, code, chat_id, name):
                user = users.get(data, chat_id)
                user["awaiting"] = "capital"
                users.save(data)
                send_text(
                    chat_id, WELCOME.format(name=name) + BAR_HINT, reply=REPLY_MENU
                )
                send_text(chat_id, render_capital(user), kb_capital())
                notify_owner(data, f"👤 کاربر جدید: <b>{name}</b> (<code>{chat_id}</code>)")
                return
        send_text(chat_id, DENIED)
        return

    admin = users.is_admin(data, chat_id)

    # ------------------------------------------------- the bottom bar buttons
    if text in MENU_TEXTS:
        if user.get("awaiting"):
            user["awaiting"] = None
            users.save(data)

        # The bar is already installed on the client, so these replies carry the
        # inline menu instead. Telegram allows only one reply_markup per message.
        if text == "📊 وضعیت":
            send_text(chat_id, account.describe(user), kb_back())
        elif text == "💰 سرمایه":
            send_text(chat_id, render_capital(user), kb_capital())
        elif text == "📉 عملکرد":
            send_text(chat_id, render_performance(), kb_back())
        else:
            send_text(chat_id, render_home(user, admin), kb_main(user, admin))
        return

    # ---------------------------------------------------- waiting for a number
    awaiting = user.get("awaiting")
    if awaiting and not text.startswith("/"):
        amount = parse_amount(text)
        if amount is None:
            send_text(
                chat_id,
                "عدد را نفهمیدم. فقط رقم بفرست، مثلاً <code>1500</code>",
                kb_back(),
            )
            return
        if awaiting == "capital":
            apply_capital(data, chat_id, amount, None)
        elif awaiting == "equity":
            apply_equity(data, chat_id, amount, None)
        elif awaiting == "risk":
            apply_risk(data, chat_id, amount, None)
        elif awaiting == "margin":
            apply_margin(data, chat_id, amount, None)
        return

    if awaiting:
        user["awaiting"] = None
        users.save(data)

    # --------------------------------------------------------------- commands
    if lowered.startswith("/start") or lowered.startswith("/menu"):
        capital = float(user.get("capital") or 0)
        if capital <= 0:
            send_text(
                chat_id,
                WELCOME.format(name=user.get("name", "کاربر")) + BAR_HINT,
                reply=REPLY_MENU,
            )
            send_text(chat_id, render_capital(user), kb_capital())
            user["awaiting"] = "capital"
            users.save(data)
            return
        send_text(chat_id, render_home(user, admin) + BAR_HINT, reply=REPLY_MENU)
        send_text(chat_id, "منوی کامل:", kb_main(user, admin))
        return

    if lowered.startswith("/capital"):
        amount = parse_amount(arg_after(text, "/capital"))
        if amount is None:
            send_text(chat_id, render_capital(user), kb_capital())
            return
        apply_capital(data, chat_id, amount, None)
        return

    if lowered.startswith("/equity"):
        amount = parse_amount(arg_after(text, "/equity"))
        if amount is None:
            send_text(chat_id, render_equity(user), kb_equity())
            return
        apply_equity(data, chat_id, amount, None)
        return

    if lowered.startswith("/risk"):
        amount = parse_amount(arg_after(text, "/risk"))
        if amount is None:
            send_text(chat_id, render_risk(user), kb_risk())
            return
        apply_risk(data, chat_id, amount, None)
        return

    if lowered.startswith("/stats") or lowered.startswith("/perf"):
        send_text(chat_id, render_performance(), kb_back())
        return

    if lowered.startswith("/help"):
        send_text(chat_id, HELP_TEXT, kb_back())
        return

    send_text(chat_id, "دکمه‌ها را بزن، یا /menu را بفرست.", kb_main(user, admin))


# ------------------------------------------------------------------ polling


def poll(seconds: int) -> int:
    _, allowed = _credentials()
    data = users.load()
    deadline = time.time() + seconds
    handled = 0

    remember_bot_username(data)
    print(
        f"polling for {seconds}s · owner {data.get('owner')} · "
        f"{len(data['users'])} user(s) · bot @{data.get('bot_username') or '?'}"
    )

    while time.time() < deadline:
        offset = int(data.get("last_update_id") or 0) + 1
        result = _call(
            "getUpdates",
            {
                "offset": offset,
                "timeout": 20,
                "allowed_updates": ["message", "callback_query"],
            },
        )
        if not result.get("ok"):
            time.sleep(2)
            continue

        for update in result.get("result", []):
            data["last_update_id"] = update["update_id"]

            if "callback_query" in update:
                handle_callback(update["callback_query"], data)
            elif "message" in update:
                handle_message(update["message"], data)

            handled += 1
            users.save(data)

    users.save(data)
    print(f"handled {handled} update(s)")
    return 0


def main(argv: list[str]) -> int:
    seconds = int(argv[1]) if len(argv) > 1 else 240
    return poll(seconds)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
