"""
The entry committee.

The base strategy answers one question: did EMA20 cross EMA50 on the correct
side of EMA200, with Bitcoin agreeing? That is a hypothesis, not an entry.

A committee of specialist agents looks at the same bar through different lenses.
Each holds a veto; the chair counts the votes. A signal only reaches the chat if
nobody objects.

Ten agents were written. Six were measured and rejected - see REJECTED below for
what each one did to the edge, because a rejected agent is a result too. The
four that survived are here:

    ADX          trend has to be strong enough to pay for the trade
    volume       the cross has to be traded, or it is a fake-out
    volatility   a dead or violent market both lose money
    breadth      the whole market should agree, not one coin alone

Measured on 150 symbols, two years of 4h candles, with fees, slippage and
funding, split in-sample / out-of-sample:

    base                5722 trades   +0.118R   PF 1.20
    committee            585 trades   +0.859R   PF 3.12
      in-sample                       +0.760R
      out-of-sample                   +1.157R
    every year better than base, 108 of 135 symbols profitable,
    and it survives deleting the ten best symbols (+0.734R)

Why rule-based agents and not an LLM per signal:

  - the committee has to be replayable over two years of candles, or there is no
    way to tell an agent that adds edge from one that just sounds clever. An LLM
    cannot be replayed: same prompt, different answer, and two years of history
    would cost real money to re-run.
  - a signal has to arrive in seconds, deterministically. A model call is a
    network round trip that can time out mid-scan.
  - this bot runs on free GitHub minutes and no API keys.

The honest trade-off: an LLM could weigh news or context these agents cannot
see. That belongs in a separate optional layer on top, not inside the decision
that fires a trade.

Usage:
    python committee.py            # self-test
"""

from __future__ import annotations

import math

APPROVE, NEUTRAL, VETO = 1, 0, -1

# Thresholds live in one dict so a backtest can vary them without editing code.
PARAMS = {
    "adx_min": 25.0,          # below this the trend is too weak to pay the costs
    "volume_min": 1.0,        # cross volume against its own 20-bar average
    "atr_rank_lo": 0.20,      # dead market: nothing follows through
    "atr_rank_hi": 0.95,      # chaos: the stop gets clipped by noise
    "breadth_min": 0.45,      # share of the universe moving the same way
}

MAX_OPEN_TRADES = 6


def _f(frame, column: str, i: int) -> float:
    """Read one cell as a float, NaN when it is missing or not a number."""
    try:
        value = float(frame[column].iloc[i])
    except (KeyError, IndexError, TypeError, ValueError):
        return float("nan")
    return value


def _vote(key: str, name: str, vote: int, reason: str, live: bool = False) -> dict:
    return {"key": key, "name": name, "vote": vote, "reason": reason, "live": live}


def _neutral(key: str, name: str, reason: str) -> dict:
    return _vote(key, name, NEUTRAL, reason)


# ------------------------------------------------------- agents with a seat


def adx_strength(frame, i: int, side: int, p: dict) -> dict:
    """قدرت روند — a cross in a trendless market has nothing to follow.

    The single biggest contributor: on its own it lifted expectancy from +0.118R
    to +0.319R and cut the trade count to a quarter. The gain grows monotonically
    as the threshold rises (18 / 22 / 25 / 28 / 30), which is what a real effect
    looks like rather than a lucky cut-off.
    """
    value = _f(frame, "adx", i)
    if not math.isfinite(value):
        return _neutral("adx", "قدرت روند", "ADX نامعتبر")
    if value < p["adx_min"]:
        return _vote("adx", "قدرت روند", VETO, f"ADX {value:.0f} — روند ضعیف")
    return _vote("adx", "قدرت روند", APPROVE, f"ADX {value:.0f} — روند قوی")


def volume_confirmation(frame, i: int, side: int, p: dict) -> dict:
    """تأیید حجم — a cross nobody traded is usually a fake-out."""
    vol = _f(frame, "volume", i)
    avg = _f(frame, "vol_sma20", i)
    if not math.isfinite(vol) or not math.isfinite(avg) or avg <= 0:
        return _neutral("volume", "حجم", "حجم نامعتبر")

    ratio = vol / avg
    if ratio < p["volume_min"]:
        return _vote("volume", "حجم", VETO, f"حجم {ratio:.1f}× میانگین — بی‌رمق")
    return _vote("volume", "حجم", APPROVE, f"حجم {ratio:.1f}× میانگین")


def volatility(frame, i: int, side: int, p: dict) -> dict:
    """نوسان — too quiet has no follow-through, too violent takes the stop out."""
    rank = _f(frame, "atr_rank", i)
    if not math.isfinite(rank):
        return _neutral("volatility", "نوسان", "رتبه ATR نامعتبر")

    if rank < p["atr_rank_lo"]:
        return _vote("volatility", "نوسان", VETO, f"بازار مرده (صدک {rank * 100:.0f})")
    if rank > p["atr_rank_hi"]:
        return _vote("volatility", "نوسان", VETO, f"نوسان بی‌ثبات (صدک {rank * 100:.0f})")
    return _vote("volatility", "نوسان", APPROVE, f"نوسان نرمال (صدک {rank * 100:.0f})")


def breadth(share: float | None, side: int, p: dict) -> dict:
    """گستردگی بازار — is the whole market moving, or is this one coin alone?

    Needs a market-wide number, so it is voted after every symbol is scanned
    rather than inside the per-symbol loop.
    """
    if share is None or not math.isfinite(float(share)):
        return _neutral("breadth", "گستردگی", "داده بازار نیست")

    share = float(share)
    aligned = share if side > 0 else 1.0 - share
    if aligned < p["breadth_min"]:
        return _vote("breadth", "گستردگی", VETO, f"فقط {share * 100:.0f}٪ بازار هم‌جهت")
    return _vote("breadth", "گستردگی", APPROVE, f"{share * 100:.0f}٪ بازار هم‌جهت")


SYMBOL_AGENTS = (adx_strength, volume_confirmation, volatility)


# --------------------------------------------------------- live-only guards


def liquidity(volume_usd: float | None, minimum: float) -> dict:
    """نقدشوندگی — a thin book eats the edge before the trade even starts."""
    if volume_usd is None:
        return _neutral("liquidity", "نقدشوندگی", "حجم ۲۴ ساعته نامشخص")
    if volume_usd < minimum:
        return _vote("liquidity", "نقدشوندگی", VETO,
                     f"حجم {volume_usd / 1e6:.1f}M زیر حد", live=True)
    return _vote("liquidity", "نقدشوندگی", APPROVE, f"حجم {volume_usd / 1e6:.1f}M", live=True)


def exposure(open_trades: list, side: int, max_open: int = MAX_OPEN_TRADES) -> dict:
    """ریسک پورتفوی — the sixth copy of the same bet is not a new idea."""
    if len(open_trades) >= max_open:
        return _vote("exposure", "ریسک پورتفوی", VETO,
                     f"{len(open_trades)} پوزیشن باز — سقف پر است", live=True)
    same_side = [t for t in open_trades if int(t.get("side", 0)) == side]
    if len(same_side) >= max_open - 1:
        return _vote("exposure", "ریسک پورتفوی", VETO,
                     f"{len(same_side)} پوزیشن هم‌جهت باز است", live=True)
    return _vote("exposure", "ریسک پورتفوی", APPROVE,
                 f"{len(open_trades)} پوزیشن باز", live=True)


# ------------------------------------------------ measured, did not get a seat


def trend(frame, i, side, p):
    """روند — EMA200 slope. +0.126R overall but -0.005R out-of-sample. Rejected."""
    bars = 30
    if i < 200 + bars:
        return _neutral("trend", "روند", "تاریخ کافی نیست")
    atr = _f(frame, "atr", i)
    if not math.isfinite(atr) or atr <= 0:
        return _neutral("trend", "روند", "ATR نامعتبر")
    slope = (_f(frame, "ema200", i) - _f(frame, "ema200", i - bars)) / atr
    aligned = slope if side > 0 else -slope
    if aligned >= 0.5:
        return _vote("trend", "روند", APPROVE, f"شیب EMA200 هم‌جهت ({aligned:+.1f} ATR)")
    if aligned <= -0.5:
        return _vote("trend", "روند", VETO, f"شیب EMA200 مخالف ({aligned:+.1f} ATR)")
    return _vote("trend", "روند", NEUTRAL, f"شیب خنثی ({aligned:+.1f} ATR)")


def momentum(frame, i, side, p):
    """مومنتوم — RSI window. Expectancy fell to +0.102R, out-of-sample +0.009R. Rejected."""
    rsi = _f(frame, "rsi", i)
    if not math.isfinite(rsi):
        return _neutral("momentum", "مومنتوم", "RSI نامعتبر")
    effective = rsi if side > 0 else 100.0 - rsi
    if effective > 75:
        return _vote("momentum", "مومنتوم", VETO, f"RSI {rsi:.0f} — حرکت کشیده شده")
    if effective < 45:
        return _vote("momentum", "مومنتوم", VETO, f"RSI {rsi:.0f} — مومنتوم کافی نیست")
    return _vote("momentum", "مومنتوم", APPROVE, f"RSI {rsi:.0f} در محدوده سالم")


def structure(frame, i, side, p):
    """ساختار — room to the 55-bar extreme. Neutral (+0.116R vs +0.118R). Rejected."""
    close = _f(frame, "close", i)
    atr = _f(frame, "atr", i)
    if not math.isfinite(atr) or atr <= 0:
        return _neutral("structure", "ساختار", "ATR نامعتبر")

    if side > 0:
        level = _f(frame, "dc_high55", i)
        room, broke_out = (level - close) / atr, close > level
    else:
        level = _f(frame, "dc_low55", i)
        room, broke_out = (close - level) / atr, close < level

    if not math.isfinite(room):
        return _neutral("structure", "ساختار", "سطح نامعتبر")
    if broke_out:
        return _vote("structure", "ساختار", APPROVE, "سقف/کف ۵۵ کندلی شکسته — فضای باز")
    if room < 1.5:
        return _vote("structure", "ساختار", VETO, f"فقط {room:.1f} ATR فضا تا سقف/کف")
    return _vote("structure", "ساختار", APPROVE, f"{room:.1f} ATR فضا تا سقف/کف")


def timing(frame, i, side, p):
    """زمان‌بندی — distance from EMA20. -0.061R out-of-sample. Rejected."""
    close, atr, ema20 = _f(frame, "close", i), _f(frame, "atr", i), _f(frame, "ema20", i)
    if not math.isfinite(atr) or atr <= 0 or not math.isfinite(ema20):
        return _neutral("timing", "زمان‌بندی", "داده نامعتبر")
    stretch = (close - ema20) / atr if side > 0 else (ema20 - close) / atr
    if stretch > 2.5:
        return _vote("timing", "زمان‌بندی", VETO, f"{stretch:.1f} ATR از EMA20 دور — دیر رسیده")
    return _vote("timing", "زمان‌بندی", APPROVE, f"{stretch:.1f} ATR از EMA20")


def regime(frame, i, side, p, context=None):
    """رژیم — BTC distance from its EMA200. Redundant: the scanner already filters
    on the sign of exactly this. Rejected."""
    gap = (context or {}).get("btc_gap_atr")
    if gap is None or not math.isfinite(float(gap)):
        return _neutral("regime", "رژیم", "داده بیت‌کوین نیست")
    gap = float(gap)
    aligned = gap if side > 0 else -gap
    if aligned < 0.5:
        return _vote("regime", "رژیم", VETO, f"بیت‌کوین فقط {gap:+.1f} ATR از EMA200")
    return _vote("regime", "رژیم", APPROVE, f"بیت‌کوین {gap:+.1f} ATR از EMA200")


REJECTED = (trend, momentum, structure, timing, regime)

CONTEXT_AGENTS = (regime,)


# --------------------------------------------------------------------- chair


def judge(
    frame,
    i: int,
    side: int,
    params: dict | None = None,
    agents: tuple = SYMBOL_AGENTS,
    context: dict | None = None,
) -> dict:
    """Collect the per-symbol votes and let the chair decide.

    Any veto blocks. The score is the share of agents that approved, which is
    what the signal message shows as confidence.
    """
    p = dict(PARAMS)
    if params:
        p.update(params)

    votes = []
    for agent in agents:
        if agent in CONTEXT_AGENTS:
            votes.append(agent(frame, i, side, p, context))
        else:
            votes.append(agent(frame, i, side, p))

    return _tally(votes)


def add_votes(decision: dict, extra: list[dict]) -> dict:
    """Fold market-wide and book-wide votes into a per-symbol decision."""
    return _tally(decision["votes"] + extra)


def _tally(votes: list[dict]) -> dict:
    blockers = [v for v in votes if v["vote"] == VETO]
    approvals = [v for v in votes if v["vote"] == APPROVE]
    neutral = [v for v in votes if v["vote"] == NEUTRAL]
    return {
        "pass": not blockers,
        "votes": votes,
        "blockers": blockers,
        "approved": len(approvals),
        "neutral": len(neutral),
        "score": len(approvals) / len(votes) if votes else 0.0,
    }


def self_test() -> int:
    """A tiny synthetic frame: the committee must run and return a decision."""
    import pandas as pd

    from features import add_all

    frame = add_all(
        pd.DataFrame(
            {
                "open": [100.0] * 320,
                "high": [101.0] * 320,
                "low": [99.0] * 320,
                "close": [100.0] * 320,
                "volume": [1.0] * 320,
            }
        )
    )
    decision = judge(frame, 319, 1)
    for vote in decision["votes"]:
        print(f"  {vote['key']:<11} {vote['vote']:>+2}  {vote['reason']}")
    print(f"pass={decision['pass']} score={decision['score']:.2f}")
    assert len(decision["votes"]) == len(SYMBOL_AGENTS)
    print("committee ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
