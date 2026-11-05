"""
OKX data source.

GitHub's runners sit in the US, where Binance answers 451 and Bybit answers 403.
OKX serves the same perpetual candles and is reachable from there, so the live
scanner reads from OKX.

Symbols are returned in Binance style (BTCUSDT) so the rest of the codebase does
not care which exchange produced the candles.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from data import NON_CRYPTO, BAD_PREFIXES
from universe import UNIVERSE

BASE = "https://www.okx.com"

# OKX bar codes differ from Binance: hours are uppercase, minutes lowercase.
BAR_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1Dutc",
}

BAR_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def _get(path: str, params: dict | None = None, retries: int = 4) -> list:
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)

    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "crypto-signal-bot/1.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("code") not in ("0", 0):
                raise RuntimeError(f"okx error {payload.get('code')}: {payload.get('msg')}")
            return payload["data"]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"okx request failed: {url} ({last})")


def to_binance_name(inst_id: str) -> str:
    return inst_id.split("-")[0] + "USDT"


def to_okx_id(symbol: str) -> str:
    return symbol[: -len("USDT")] + "-USDT-SWAP"


STABLECOINS = {"USDC", "USDT", "DAI", "FDUSD", "TUSD", "BUSD", "USDD", "PYUSD", "USDE"}


def top_symbols(limit: int | None = None) -> list[str]:
    """The universe, filtered down to what OKX is actually trading right now."""
    instruments = _get("/api/v5/public/instruments", {"instType": "SWAP"})
    live = {
        item["instId"].split("-")[0]
        for item in instruments
        if item.get("instId", "").endswith("-USDT-SWAP") and item.get("state") == "live"
    }
    symbols = [
        f"{base}USDT"
        for base in UNIVERSE
        if base in live and base not in STABLECOINS
    ]
    return symbols[:limit] if limit else symbols


def volumes() -> dict[str, float]:
    """24h turnover in USD for every USDT perpetual, keyed by base asset."""
    tickers = _get("/api/v5/market/tickers", {"instType": "SWAP"})
    result: dict[str, float] = {}
    for ticker in tickers:
        inst_id = ticker.get("instId", "")
        if not inst_id.endswith("-USDT-SWAP"):
            continue
        try:
            result[inst_id.split("-")[0]] = float(ticker["volCcy24h"]) * float(ticker["last"])
        except (KeyError, TypeError, ValueError):
            continue
    return result


def fetch_klines(symbol: str, interval: str, days: int) -> list[list]:
    """Return candles in Binance row format so downstream code stays unchanged."""
    if interval not in BAR_MAP:
        raise ValueError(f"unsupported interval: {interval}")

    bar = BAR_MAP[interval]
    step = BAR_MS[interval]
    wanted = int(days * 86_400_000 / step)
    inst_id = to_okx_id(symbol)

    rows: list[list] = []
    cursor: str | None = None

    while len(rows) < wanted + 2:
        params = {"instId": inst_id, "bar": bar, "limit": 300}
        if cursor:
            params["after"] = cursor
        batch = _get("/api/v5/market/candles", params)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0]
        if len(batch) < 300:
            break
        time.sleep(0.15)

    # OKX returns newest first and marks the still-forming candle with confirm=0.
    closed = [row for row in rows if row[8] == "1"]
    closed.sort(key=lambda row: int(row[0]))

    if len(closed) > wanted:
        closed = closed[-wanted:]

    return [
        [int(row[0]), row[1], row[2], row[3], row[4], row[5], int(row[0]) + step, row[7], 0]
        for row in closed
    ]
