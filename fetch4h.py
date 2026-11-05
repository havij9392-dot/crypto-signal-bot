"""
Download deep 4h history for the research universe.

OKX's /market/candles endpoint only returns the most recent ~1440 bars, which is
240 days on 4h. Two years of data needs /market/history-candles, paginated
backwards with the `after` cursor.

Writes one CSV per symbol into data4h/, in the same column layout the rest of
the project already reads.

Usage:
    python fetch4h.py 150          # top 150 symbols by 24h turnover
    python fetch4h.py 150 730      # explicit lookback in days
"""

from __future__ import annotations

import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import json

from okx import _get, to_okx_id, top_symbols, volumes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data4h")

BAR = "4H"
STEP_MS = 14_400_000
PAGE = 100          # history-candles caps at 100 rows
WORKERS = 12


def fetch_one(symbol: str, days: int) -> tuple[str, int]:
    wanted = int(days * 86_400_000 / STEP_MS)
    inst_id = to_okx_id(symbol)

    rows: list[list] = []
    cursor: str | None = None
    guard = 0

    while len(rows) < wanted + 2 and guard < 200:
        guard += 1
        params = {"instId": inst_id, "bar": BAR, "limit": PAGE}
        if cursor:
            params["after"] = cursor
        try:
            batch = _get("/api/v5/market/history-candles", params)
        except RuntimeError:
            break
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0]
        if len(batch) < PAGE:
            break
        time.sleep(0.05)

    closed = [row for row in rows if row[8] == "1"]
    closed.sort(key=lambda row: int(row[0]))
    if len(closed) > wanted:
        closed = closed[-wanted:]
    if not closed:
        raise RuntimeError("no closed candles returned")

    path = os.path.join(OUT_DIR, f"{symbol}_4h.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("open_time_ms,open_time,open,high,low,close,volume,quote_volume,trades\n")
        for row in closed:
            stamp = int(row[0])
            readable = time.strftime("%Y-%m-%d %H:%M", time.gmtime(stamp / 1000))
            handle.write(
                f"{stamp},{readable},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},0,0\n"
            )

    return symbol, len(closed)


def main(argv: list[str]) -> int:
    count = int(argv[1]) if len(argv) > 1 else 150
    days = int(argv[2]) if len(argv) > 2 else 730

    os.makedirs(OUT_DIR, exist_ok=True)

    print("ranking symbols by 24h turnover...")
    turnover = volumes()
    live = top_symbols()
    live.sort(key=lambda s: -turnover.get(s[: -len("USDT")], 0))
    symbols = live[:count]
    print(f"{len(symbols)} symbols selected (of {len(live)} live), target {days} days each")

    ok, failed = 0, []
    started = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, s, days): s for s in symbols}
        for index, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                _, bars = future.result()
                ok += 1
                if index % 20 == 0 or index == len(symbols):
                    print(f"  {index}/{len(symbols)} done · last {symbol} {bars} bars")
            except Exception as exc:  # noqa: BLE001
                failed.append(symbol)
                print(f"  {symbol} failed: {exc}")

    elapsed = time.time() - started
    print(f"\n{ok} symbols written to data4h/ in {elapsed:.0f}s, {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed[:15]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
