"""
Funding-rate history.

Binance pays funding every 8 hours and keeps the full history for free. It is the
only public series that shows how crowded each side of the book is, which is why
it is worth building a strategy around.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = "https://fapi.binance.com"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FUNDING_DIR = os.path.join(DATA_DIR, "funding")


def _get(path: str, params: dict, retries: int = 4) -> list:
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "crypto-signal-bot/1.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"funding request failed: {url} ({last})")


def fetch(symbol: str, days: int) -> list[list]:
    end_ms = int(time.time() * 1000)
    cursor = end_ms - days * 86_400_000
    rows: list[list] = []

    while cursor < end_ms:
        batch = _get(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)

    return rows


def main(argv: list[str]) -> int:
    days = int(argv[1]) if len(argv) > 1 else 365
    os.makedirs(FUNDING_DIR, exist_ok=True)

    symbols = sorted(
        os.path.basename(p).split("_")[0]
        for p in glob.glob(os.path.join(DATA_DIR, "*_15m.csv"))
    )
    if not symbols:
        print("no 15m data yet")
        return 1

    for index, symbol in enumerate(symbols, start=1):
        try:
            rows = fetch(symbol, days)
            path = os.path.join(FUNDING_DIR, f"{symbol}.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["funding_time_ms", "funding_time", "funding_rate", "mark_price"])
                for row in rows:
                    stamp = datetime.fromtimestamp(
                        int(row["fundingTime"]) / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M")
                    writer.writerow(
                        [row["fundingTime"], stamp, row["fundingRate"], row["markPrice"]]
                    )
            print(f"[{index:2}/{len(symbols)}] {symbol:12} {len(rows):5} funding records")
        except Exception as exc:  # noqa: BLE001
            print(f"[{index:2}/{len(symbols)}] {symbol:12} FAILED: {exc}")

    print(f"done - files in {FUNDING_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
