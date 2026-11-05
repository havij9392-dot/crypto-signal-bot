"""
Phase 1 - Data engine.

Pulls USDT-M futures candles from Binance's public API (no API key needed)
and stores them as CSV in ./data.

Deliberately stdlib-only: no pip install, so it runs anywhere, including
GitHub Actions, in a few seconds.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = "https://fapi.binance.com"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

INTERVAL_MS = {
    "1m": 60_000,
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

# Leveraged tokens and other junk we never want to trade.
BAD_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
BAD_PREFIXES = ("1000",)  # 1000SHIB-style contracts confuse the signal message

# Binance lists tokenised equities and commodities on the same futures API
# (GOOGLUSDT, XAUUSDT, CLUSDT...). They are not crypto and have short history,
# so they must never reach the backtester.
NON_CRYPTO = {
    "SOXL", "SOXS", "SNDK", "MU", "KORU", "SKHYNIX", "SKHY", "SNXX", "SPCX",
    "GOOGL", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "NFLX", "AMD",
    "INTC", "COIN", "MSTR", "PLTR", "CRCL", "BABA", "TSM", "ORCL", "IBM",
    "XAU", "XAG", "XPT", "XPD", "CL", "BZ", "NG", "COPPER",
}

CSV_HEADER = [
    "open_time_ms",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
]


def _get(path: str, params: dict | None = None, retries: int = 4) -> object:
    """GET a Binance endpoint with simple retry/backoff."""
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "crypto-signal-bot/1.0"}
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - network layer, we retry everything
            last_error = exc
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"request failed after {retries} tries: {url} ({last_error})")


def top_symbols(
    limit: int = 20,
    min_quote_volume: float = 50_000_000,
    min_history_days: int = 730,
) -> list[str]:
    """Pick the most liquid USDT perpetuals that have enough history to backtest.

    Requires a real perpetual contract, real crypto, and a listing older than
    `min_history_days` - otherwise a symbol would win the backtest on 3 months
    of lucky data.
    """
    info = _get("/fapi/v1/exchangeInfo")
    assert isinstance(info, dict)
    now_ms = int(time.time() * 1000)
    oldest_allowed = now_ms - min_history_days * 86_400_000

    eligible: set[str] = set()
    for contract in info["symbols"]:
        symbol = contract["symbol"]
        if not symbol.endswith("USDT"):
            continue
        if contract.get("contractType") != "PERPETUAL":
            continue
        if contract.get("status") != "TRADING":
            continue
        if symbol.endswith(BAD_SUFFIXES) or symbol.startswith(BAD_PREFIXES):
            continue
        if symbol[: -len("USDT")] in NON_CRYPTO:
            continue
        if int(contract.get("onboardDate", 0)) > oldest_allowed:
            continue
        eligible.add(symbol)

    tickers = _get("/fapi/v1/ticker/24hr")
    assert isinstance(tickers, list)

    rows = []
    for ticker in tickers:
        symbol = ticker["symbol"]
        if symbol not in eligible:
            continue
        try:
            turnover = float(ticker["quoteVolume"])
        except (KeyError, TypeError, ValueError):
            continue
        if turnover < min_quote_volume:
            continue
        rows.append((turnover, symbol))

    rows.sort(reverse=True)
    return [symbol for _, symbol in rows[:limit]]


def fetch_klines(symbol: str, interval: str, days: int) -> list[list]:
    """Download `days` worth of candles, paginating around Binance's 1500 cap."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval: {interval}")

    step = INTERVAL_MS[interval]
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 86_400_000

    candles: list[list] = []
    cursor = start_ms
    while cursor < now_ms:
        batch = _get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": 1500},
        )
        if not batch:
            break
        candles.extend(batch)
        last_open = int(batch[-1][0])
        if len(batch) < 1500:
            break
        cursor = last_open + step
        time.sleep(0.25)

    # Drop the still-forming last candle so every row is a closed candle.
    if candles and int(candles[-1][6]) > now_ms:
        candles.pop()

    return candles


def save_csv(symbol: str, interval: str, candles: list[list]) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for candle in candles:
            open_ms = int(candle[0])
            stamp = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
            writer.writerow(
                [
                    open_ms,
                    stamp,
                    candle[1],
                    candle[2],
                    candle[3],
                    candle[4],
                    candle[5],
                    candle[7],
                    candle[8],
                ]
            )
    return path


def load_csv(path: str) -> list[dict]:
    """Read a saved CSV back as plain dicts with floats. Used by the backtester."""
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            {
                "open_time_ms": int(row["open_time_ms"]),
                "open_time": row["open_time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "quote_volume": float(row["quote_volume"]),
                "trades": int(row["trades"]),
            }
            for row in csv.DictReader(handle)
        ]


def health_check() -> None:
    price = _get("/fapi/v1/ticker/price", {"symbol": "BTCUSDT"})
    print(f"connection ok - BTCUSDT = {price['price']}")


def main(argv: list[str]) -> int:
    interval = argv[1] if len(argv) > 1 else "4h"
    days = int(argv[2]) if len(argv) > 2 else 730
    wanted = argv[3:] or None

    health_check()

    symbols = wanted or top_symbols(limit=20)
    print(f"interval={interval} days={days} symbols={len(symbols)}")

    manifest = []
    for index, symbol in enumerate(symbols, start=1):
        try:
            candles = fetch_klines(symbol, interval, days)
            path = save_csv(symbol, interval, candles)
            first = candles[0][0] if candles else 0
            last = candles[-1][0] if candles else 0
            span_days = round((last - first) / 86_400_000, 1) if candles else 0
            print(f"[{index:2}/{len(symbols)}] {symbol:14} {len(candles):5} candles  {span_days:6} days")
            manifest.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "candles": len(candles),
                    "first": datetime.fromtimestamp(first / 1000, tz=timezone.utc).isoformat()
                    if candles
                    else None,
                    "last": datetime.fromtimestamp(last / 1000, tz=timezone.utc).isoformat()
                    if candles
                    else None,
                    "file": os.path.basename(path),
                }
            )
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the run
            print(f"[{index:2}/{len(symbols)}] {symbol:14} FAILED: {exc}")

    with open(os.path.join(DATA_DIR, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "interval": interval,
                "requested_days": days,
                "symbols": manifest,
            },
            handle,
            indent=2,
        )

    print(f"done - {len(manifest)} files in {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
