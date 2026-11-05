"""
Multi-user registry.

The bot used to serve exactly one chat (TELEGRAM_CHAT_ID). It now serves a small
private group: the owner mints an invite link, and whoever opens it gets their
own capital, risk and signal settings.

Storage is state/users.json, committed back by the workflow - a git file is the
only free persistence this project has. It holds Telegram chat ids, nothing
secret; the bot token never leaves GitHub Secrets.

Shape:

    {
      "owner": "6090974409",
      "bot_username": "zakaisignal_bot",
      "last_update_id": 408113309,
      "users": {
        "6090974409": {
          "name": "امیر", "capital": 1000.0, "risk_pct": 1.0, "peak": 1000.0,
          "signals": true, "awaiting": null, "joined": "2026-11-05T19:00:00Z"
        }
      },
      "invites": {"ZAK7F2A9": {"by": "6090974409", "created": "...", "used_by": null}}
    }
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, "state")
USERS_FILE = os.path.join(STATE_DIR, "users.json")
LEGACY_ACCOUNT = os.path.join(STATE_DIR, "account.json")

INVITE_TTL = timedelta(hours=72)
MAX_USERS = 25


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def owner_id() -> str:
    """The owner's chat id, from the environment (GitHub Secret) or a local .env."""
    if not os.environ.get("TELEGRAM_CHAT_ID"):
        from notify import load_env_file

        load_env_file()
    return os.environ.get("TELEGRAM_CHAT_ID", "").strip()


def new_user(name: str, capital: float = 0.0, risk_pct: float = 1.0) -> dict:
    return {
        "name": name or "کاربر",
        "capital": round(float(capital), 2),
        "risk_pct": round(float(risk_pct), 2),
        "peak": round(float(capital), 2),
        "signals": True,
        "awaiting": None,
        "joined": _now(),
    }


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def _migrate(data: dict) -> None:
    """Seed the owner from the single-user state/account.json, once."""
    if data["users"] or not data["owner"]:
        return

    legacy = _read_json(LEGACY_ACCOUNT)
    data["users"][data["owner"]] = new_user(
        "امیر",
        capital=float(legacy.get("capital") or 0),
        risk_pct=float(legacy.get("risk_pct") or 1.0),
    )
    data["users"][data["owner"]]["peak"] = float(legacy.get("peak") or 0)
    data["last_update_id"] = int(legacy.get("last_update_id") or 0)


def load() -> dict:
    data = _read_json(USERS_FILE)
    data.setdefault("owner", owner_id())
    data.setdefault("bot_username", "")
    data.setdefault("last_update_id", 0)
    data.setdefault("users", {})
    data.setdefault("invites", {})
    if not data["owner"]:
        data["owner"] = owner_id()
    _migrate(data)
    return data


def save(data: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, USERS_FILE)


def is_admin(data: dict, chat_id: str) -> bool:
    return str(chat_id) == str(data.get("owner") or "")


def get(data: dict, chat_id: str) -> dict | None:
    return data["users"].get(str(chat_id))


def subscribers(data: dict) -> list[tuple[str, dict]]:
    """Everyone who should receive signals, owner first."""
    out = []
    for chat_id, user in data["users"].items():
        if user.get("signals", True):
            out.append((chat_id, user))
    out.sort(key=lambda pair: (pair[0] != str(data.get("owner")), pair[1].get("name", "")))
    return out


def count(data: dict) -> int:
    return len(data["users"])


def add_user(data: dict, chat_id: str, name: str) -> dict:
    user = new_user(name)
    data["users"][str(chat_id)] = user
    return user


def remove_user(data: dict, chat_id: str) -> bool:
    chat_id = str(chat_id)
    if chat_id == str(data.get("owner") or ""):
        return False  # the owner cannot remove himself
    return data["users"].pop(chat_id, None) is not None


def _prune_invites(data: dict) -> None:
    cutoff = datetime.now(timezone.utc) - INVITE_TTL
    dead = []
    for code, meta in data["invites"].items():
        try:
            created = datetime.strptime(meta["created"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except (KeyError, ValueError):
            dead.append(code)
            continue
        if created < cutoff or meta.get("used_by"):
            dead.append(code)
    for code in dead:
        data["invites"].pop(code, None)


def make_invite(data: dict, by: str) -> str:
    _prune_invites(data)
    code = secrets.token_hex(4).upper()
    while code in data["invites"]:
        code = secrets.token_hex(4).upper()
    data["invites"][code] = {"by": str(by), "created": _now(), "used_by": None}
    return code


def redeem(data: dict, code: str, chat_id: str, name: str) -> bool:
    _prune_invites(data)
    code = (code or "").strip().upper()
    meta = data["invites"].get(code)
    if not meta or meta.get("used_by"):
        return False
    if str(chat_id) in data["users"]:
        return True
    if count(data) >= MAX_USERS:
        return False

    data["users"][str(chat_id)] = new_user(name)
    meta["used_by"] = str(chat_id)
    meta["used_at"] = _now()
    return True


def invite_link(data: dict, code: str) -> str:
    username = data.get("bot_username") or "zakaisignal_bot"
    return f"https://t.me/{username}?start={code}"
