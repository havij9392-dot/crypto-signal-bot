"""
Phase 4 - Telegram transport.

Reads credentials from environment variables first (that is how GitHub Actions
injects secrets) and falls back to a local .env file for development.

Usage:
    python notify.py "some text"
    python notify.py --demo        # sends a sample signal to check formatting
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(HERE, ".env")

API = "https://api.telegram.org/bot{token}/{method}"


def load_env_file(path: str = ENV_FILE) -> None:
    """Load KEY=VALUE lines into os.environ without overriding real env vars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _credentials() -> tuple[str, str]:
    load_env_file()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing - set them in .env locally "
            "or as repository secrets on GitHub"
        )
    return token, chat_id


def send(text: str, chat_id: str | None = None, silent: bool = False) -> bool:
    """Send one message. Returns True on success, prints the reason on failure."""
    token, default_chat = _credentials()
    target = chat_id or default_chat

    payload = urllib.parse.urlencode(
        {
            "chat_id": target,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
            "disable_notification": "true" if silent else "false",
        }
    ).encode()

    for attempt in range(3):
        try:
            request = urllib.request.Request(API.format(token=token, method="sendMessage"), data=payload)
            with urllib.request.urlopen(request, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))
            if result.get("ok"):
                return True
            print(f"telegram rejected the message: {result.get('description')}")
            return False
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"telegram HTTP {exc.code}: {body}")
            if exc.code in (400, 401, 403):
                return False  # retrying will not help
        except Exception as exc:  # noqa: BLE001 - network hiccup, retry
            print(f"send attempt {attempt + 1} failed: {exc}")
        time.sleep(2 * (attempt + 1))

    return False


DEMO = """🔔 <b>سیگنال آزمایشی</b>
───────────────
<b>جفت‌ارز:</b> BTCUSDT
<b>جهت:</b> خرید (لانگ)
<b>ورود:</b> 76,900
<b>حد ضرر:</b> 74,200
<b>حد سود:</b> 83,300
<b>اهرم پیشنهادی:</b> ۲x
───────────────
این پیام فقط برای آزمایش مسیر ارسال است.
هیچ معامله‌ای انجام نده."""


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print('usage: python notify.py "message"   |   python notify.py --demo')
        return 1

    text = DEMO if argv[1] == "--demo" else " ".join(argv[1:])
    ok = send(text)
    print("sent" if ok else "failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
