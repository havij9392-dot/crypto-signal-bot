"""
Fifty entry strategies, each returning +1 (long), -1 (short) or 0 per bar.

Sources are named where a strategy belongs to a specific trader or author. The
rest are standard families (breakout, mean reversion, trend, pattern) that any
desk would recognise. All of them read closed bars only.

The point of this file is to let the sweep in sweep.py judge them on identical
terms: same cost model, same fills, same in-sample / out-of-sample split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import heikin_ashi, swing_points


def _sig(long_: np.ndarray | pd.Series, short: np.ndarray | pd.Series, index) -> pd.Series:
    long_ = np.asarray(long_, dtype=bool)
    short = np.asarray(short, dtype=bool)
    return pd.Series(np.where(long_, 1, np.where(short, -1, 0)), index=index)


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_dn(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


# ---------------------------------------------------------------------------
# 1-8  moving-average and momentum trend
# ---------------------------------------------------------------------------
def ema_cross_20_50(df):
    f, s = df["ema20"], df["ema50"]
    return _sig(_cross_up(f, s) & (df["close"] > df["ema200"]),
                _cross_dn(f, s) & (df["close"] < df["ema200"]), df.index)


def ema_cross_9_21(df):
    f, s = df["ema9"], df["ema21"]
    return _sig(_cross_up(f, s) & (df["close"] > df["ema200"]),
                _cross_dn(f, s) & (df["close"] < df["ema200"]), df.index)


def ema_cross_13_34(df):
    f, s = df["ema13"], df["ema34"]
    return _sig(_cross_up(f, s), _cross_dn(f, s), df.index)


def guppy_mma(df):
    """Daryl Guppy's multiple moving average ribbon."""
    short_group = (df["ema9"] + df["ema13"] + df["ema20"]) / 3
    long_group = (df["ema34"] + df["ema50"] + df["ema100"]) / 3
    return _sig(_cross_up(short_group, long_group), _cross_dn(short_group, long_group), df.index)


def macd_cross(df):
    """Gerald Appel's MACD."""
    return _sig(_cross_up(df["macd"], df["macd_signal"]) & (df["macd"] < 0),
                _cross_dn(df["macd"], df["macd_signal"]) & (df["macd"] > 0), df.index)


def macd_hist_flip(df):
    h = df["macd_hist"]
    return _sig((h > 0) & (h.shift(1) <= 0), (h < 0) & (h.shift(1) >= 0), df.index)


def elder_triple_screen(df):
    """Alexander Elder: trend on the slow screen, momentum on the fast one."""
    trend_up = df["ema13"] > df["ema13"].shift(1)
    trend_dn = df["ema13"] < df["ema13"].shift(1)
    up = trend_up & (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
    dn = trend_dn & (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)
    return _sig(up, dn, df.index)


def elder_impulse(df):
    """Elder's Impulse: EMA13 slope plus MACD histogram, both aligned."""
    up = (df["ema13"] > df["ema13"].shift(1)) & (df["macd_hist"] > df["macd_hist"].shift(1))
    dn = (df["ema13"] < df["ema13"].shift(1)) & (df["macd_hist"] < df["macd_hist"].shift(1))
    fresh_up = up & ~up.shift(1).fillna(False)
    fresh_dn = dn & ~dn.shift(1).fillna(False)
    return _sig(fresh_up, fresh_dn, df.index)


# ---------------------------------------------------------------------------
# 9-18  oscillators and mean reversion
# ---------------------------------------------------------------------------
def connors_rsi2(df):
    """Larry Connors' RSI(2) pullback: buy deep dips inside an uptrend."""
    up = df["close"] > df["sma200"]
    dn = df["close"] < df["sma200"]
    return _sig(up & (df["rsi2"] < 5), dn & (df["rsi2"] > 95), df.index)


def connors_rsi2_exit_trend(df):
    up = (df["close"] > df["sma200"]) & (df["sma20"] > df["sma50"])
    dn = (df["close"] < df["sma200"]) & (df["sma20"] < df["sma50"])
    return _sig(up & (df["rsi2"] < 10), dn & (df["rsi2"] > 90), df.index)


def rsi14_pullback(df):
    up = df["close"] > df["ema200"]
    dn = df["close"] < df["ema200"]
    return _sig(up & (df["rsi"] < 40) & (df["rsi"].shift(1) >= 40),
                dn & (df["rsi"] > 60) & (df["rsi"].shift(1) <= 60), df.index)


def rsi_divergence(df):
    """Wilder-style divergence: price makes a lower low, RSI does not."""
    low10 = df["low"].rolling(10).min()
    high10 = df["high"].rolling(10).max()
    rsi_min = df["rsi"].rolling(10).min()
    rsi_max = df["rsi"].rolling(10).max()
    bull = (df["low"] <= low10) & (df["rsi"] > rsi_min + 3)
    bear = (df["high"] >= high10) & (df["rsi"] < rsi_max - 3)
    return _sig(bull & (df["rsi"] < 40), bear & (df["rsi"] > 60), df.index)


def stoch_cross(df):
    """George Lane's stochastic."""
    up = _cross_up(df["stoch_k"], df["stoch_d"]) & (df["stoch_k"] < 25)
    dn = _cross_dn(df["stoch_k"], df["stoch_d"]) & (df["stoch_k"] > 75)
    return _sig(up, dn, df.index)


def stoch_reversal(df):
    up = (df["stoch_k"] < 20) & (df["stoch_k"] > df["stoch_k"].shift(1))
    dn = (df["stoch_k"] > 80) & (df["stoch_k"] < df["stoch_k"].shift(1))
    return _sig(up, dn, df.index)


def cci_zero_cross(df):
    """Donald Lambert's CCI."""
    return _sig(_cross_up(df["cci"], pd.Series(0.0, index=df.index)),
                _cross_dn(df["cci"], pd.Series(0.0, index=df.index)), df.index)


def cci_extreme_fade(df):
    return _sig((df["cci"] < -200) & (df["cci"] > df["cci"].shift(1)),
                (df["cci"] > 200) & (df["cci"] < df["cci"].shift(1)), df.index)


def bollinger_breakout(df):
    """John Bollinger: trade the band break after a squeeze."""
    squeeze = df["bb_width_rank"] < 0.30
    up = (df["close"] > df["bb_up"]) & squeeze.shift(1).fillna(False)
    dn = (df["close"] < df["bb_dn"]) & squeeze.shift(1).fillna(False)
    return _sig(up, dn, df.index)


def bollinger_bounce(df):
    up = (df["low"] <= df["bb_dn"]) & (df["close"] > df["bb_dn"]) & (df["close"] > df["ema200"])
    dn = (df["high"] >= df["bb_up"]) & (df["close"] < df["bb_up"]) & (df["close"] < df["ema200"])
    return _sig(up, dn, df.index)


def bollinger_pctb_reversal(df):
    up = (df["bb_pctb"] < 0.05) & (df["bb_pctb"] > df["bb_pctb"].shift(1))
    dn = (df["bb_pctb"] > 0.95) & (df["bb_pctb"] < df["bb_pctb"].shift(1))
    return _sig(up, dn, df.index)


def keltner_breakout(df):
    """Chester Keltner's channel breakout."""
    return _sig((df["close"] > df["kc_up"]) & (df["close"].shift(1) <= df["kc_up"].shift(1)),
                (df["close"] < df["kc_dn"]) & (df["close"].shift(1) >= df["kc_dn"].shift(1)),
                df.index)


def keltner_bounce(df):
    up = (df["low"] <= df["kc_dn"]) & (df["close"] > df["kc_dn"])
    dn = (df["high"] >= df["kc_up"]) & (df["close"] < df["kc_up"])
    return _sig(up, dn, df.index)


def zscore_reversion(df):
    up = (df["zscore20"] < -2) & (df["zscore20"] > df["zscore20"].shift(1))
    dn = (df["zscore20"] > 2) & (df["zscore20"] < df["zscore20"].shift(1))
    return _sig(up, dn, df.index)


# ---------------------------------------------------------------------------
# 19-28  breakout, structure and the famous discretionary playbooks
# ---------------------------------------------------------------------------
def turtle_donchian20(df):
    """Richard Dennis / the Turtles, system 1."""
    return _sig(df["close"] > df["dc_high20"], df["close"] < df["dc_low20"], df.index)


def turtle_donchian55(df):
    """The Turtles' system 2, the slower one."""
    return _sig(df["close"] > df["dc_high55"], df["close"] < df["dc_low55"], df.index)


def turtle_soup(df):
    """Linda Raschke's fade of a failed 20-bar breakout."""
    up = (df["low"] < df["dc_low20"].shift(1)) & (df["close"] > df["dc_low20"].shift(1))
    dn = (df["high"] > df["dc_high20"].shift(1)) & (df["close"] < df["dc_high20"].shift(1))
    return _sig(up, dn, df.index)


def darvas_box(df):
    """Nicolas Darvas: break out of a tight box, confirmed by volume."""
    box_high = df["high"].rolling(20).max().shift(1)
    box_low = df["low"].rolling(20).min().shift(1)
    tight = (box_high - box_low) / df["close"] < 0.12
    up = (df["close"] > box_high) & tight & (df["vol_ratio"] > 1.2)
    dn = (df["close"] < box_low) & tight & (df["vol_ratio"] > 1.2)
    return _sig(up, dn, df.index)


def weinstein_stage2(df):
    """Stan Weinstein's stage analysis: only trade stage 2 advances."""
    rising = df["sma200"] > df["sma200"].shift(10)
    falling = df["sma200"] < df["sma200"].shift(10)
    up = rising & _cross_up(df["close"], df["sma200"])
    dn = falling & _cross_dn(df["close"], df["sma200"])
    return _sig(up, dn, df.index)


def minervini_trend_template(df):
    """Mark Minervini's trend template, condensed to the price conditions."""
    stacked = (df["close"] > df["sma50"]) & (df["sma50"] > df["sma200"])
    stacked_dn = (df["close"] < df["sma50"]) & (df["sma50"] < df["sma200"])
    near_high = df["close"] > df["high"].rolling(60).max().shift(1) * 0.97
    near_low = df["close"] < df["low"].rolling(60).min().shift(1) * 1.03
    up = stacked & near_high & (df["close"] > df["close"].shift(20))
    dn = stacked_dn & near_low & (df["close"] < df["close"].shift(20))
    return _sig(up, dn, df.index)


def oneil_breakout(df):
    """William O'Neil's CANSLIM base breakout on heavy volume."""
    base_high = df["high"].rolling(40).max().shift(1)
    base_low = df["low"].rolling(40).min().shift(1)
    up = (df["close"] > base_high) & (df["vol_ratio"] > 1.5) & (df["close"] > df["sma200"])
    dn = (df["close"] < base_low) & (df["vol_ratio"] > 1.5) & (df["close"] < df["sma200"])
    return _sig(up, dn, df.index)


def livermore_pivot(df):
    """Jesse Livermore's pivotal point: break of the last confirmed swing."""
    ph, pl = swing_points(df, 3)
    up = (df["close"] > ph) & (df["close"].shift(1) <= ph.shift(1))
    dn = (df["close"] < pl) & (df["close"].shift(1) >= pl.shift(1))
    return _sig(up, dn, df.index)


def wyckoff_spring(df):
    """Wyckoff accumulation: a spring below support that snaps back."""
    support = df["low"].rolling(30).min().shift(1)
    up = (df["low"] < support) & (df["close"] > support) & (df["vol_ratio"] > 1.3)
    resistance = df["high"].rolling(30).max().shift(1)
    dn = (df["high"] > resistance) & (df["close"] < resistance) & (df["vol_ratio"] > 1.3)
    return _sig(up, dn, df.index)


def market_structure(df):
    """Break of structure: higher high after a higher low."""
    ph, pl = swing_points(df, 3)
    up = (df["close"] > ph) & (pl > pl.shift(5))
    dn = (df["close"] < pl) & (ph < ph.shift(5))
    return _sig(up, dn, df.index)


def bos_retest(df):
    """Break of structure, then enter on the retest of the broken level."""
    ph, pl = swing_points(df, 3)
    broke_up = (df["close"].shift(1) > ph.shift(1)) & (df["close"].shift(2) <= ph.shift(2))
    touched = (df["low"] <= ph) & (df["close"] > ph)
    up = broke_up.shift(1).fillna(False) | (touched & broke_up.shift(2).fillna(False))
    broke_dn = (df["close"].shift(1) < pl.shift(1)) & (df["close"].shift(2) >= pl.shift(2))
    touched_dn = (df["high"] >= pl) & (df["close"] < pl)
    dn = broke_dn.shift(1).fillna(False) | (touched_dn & broke_dn.shift(2).fillna(False))
    return _sig(up, dn, df.index)


# ---------------------------------------------------------------------------
# 29-36  trend-following indicators
# ---------------------------------------------------------------------------
def adx_trend(df):
    """J. Welles Wilder's ADX, used as a trend strength gate."""
    strong = df["adx"] > 25
    up = strong & (df["plus_di"] > df["minus_di"]) & (df["plus_di"].shift(1) <= df["minus_di"].shift(1))
    dn = strong & (df["minus_di"] > df["plus_di"]) & (df["minus_di"].shift(1) <= df["plus_di"].shift(1))
    return _sig(up, dn, df.index)


def adx_rising_trend(df):
    up = (df["adx"] > 20) & (df["adx"] > df["adx"].shift(1)) & (df["plus_di"] > df["minus_di"])
    dn = (df["adx"] > 20) & (df["adx"] > df["adx"].shift(1)) & (df["minus_di"] > df["plus_di"])
    return _sig(up & ~up.shift(1).fillna(False), dn & ~dn.shift(1).fillna(False), df.index)


def supertrend_flip(df):
    d = df["st_dir"]
    return _sig((d > 0) & (d.shift(1) <= 0), (d < 0) & (d.shift(1) >= 0), df.index)


def psar_flip(df):
    """Wilder's parabolic SAR - psar_bull is 1/0, not +1/-1."""
    d = df["psar_bull"]
    return _sig((d > 0) & (d.shift(1) == 0), (d == 0) & (d.shift(1) > 0), df.index)


def ichimoku_cloud(df):
    """Goichi Hosoda's Ichimoku Kinko Hyo."""
    top = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).max(axis=1)
    bottom = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).min(axis=1)
    up = (df["close"] > top) & (df["close"].shift(1) <= top.shift(1))
    dn = (df["close"] < bottom) & (df["close"].shift(1) >= bottom.shift(1))
    return _sig(up, dn, df.index)


def ichimoku_tk(df):
    up = _cross_up(df["tenkan"], df["kijun"]) & (df["close"] > df["senkou_a"])
    dn = _cross_dn(df["tenkan"], df["kijun"]) & (df["close"] < df["senkou_a"])
    return _sig(up, dn, df.index)


def heikin_ashi_trend(df):
    ha = heikin_ashi(df)
    green = ha["close"] > ha["open"]
    red = ha["close"] < ha["open"]
    return _sig(green & ~green.shift(1).fillna(False) & (df["close"] > df["ema200"]),
                red & ~red.shift(1).fillna(False) & (df["close"] < df["ema200"]), df.index)


def aroon_trend(df):
    up = (df["aroon_up"] > 70) & (df["aroon_dn"] < 30) & (df["aroon_up"] > df["aroon_up"].shift(1))
    dn = (df["aroon_dn"] > 70) & (df["aroon_up"] < 30) & (df["aroon_dn"] > df["aroon_dn"].shift(1))
    return _sig(up & ~up.shift(1).fillna(False), dn & ~dn.shift(1).fillna(False), df.index)


def slope_momentum(df):
    up = (df["slope20"] > 0) & (df["slope20"].shift(1) <= 0) & (df["close"] > df["sma50"])
    dn = (df["slope20"] < 0) & (df["slope20"].shift(1) >= 0) & (df["close"] < df["sma50"])
    return _sig(up, dn, df.index)


# ---------------------------------------------------------------------------
# 37-46  candles, patterns and volatility
# ---------------------------------------------------------------------------
def inside_bar_breakout(df):
    inside = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    up = inside.shift(1).fillna(False) & (df["close"] > df["high"].shift(2))
    dn = inside.shift(1).fillna(False) & (df["close"] < df["low"].shift(2))
    return _sig(up, dn, df.index)


def engulfing_pattern(df):
    prev_bear = df["close"].shift(1) < df["open"].shift(1)
    prev_bull = df["close"].shift(1) > df["open"].shift(1)
    up = prev_bear & (df["close"] > df["open"]) & (df["close"] > df["open"].shift(1)) & (
        df["open"] < df["close"].shift(1)
    )
    dn = prev_bull & (df["close"] < df["open"]) & (df["close"] < df["open"].shift(1)) & (
        df["open"] > df["close"].shift(1)
    )
    return _sig(up, dn, df.index)


def pin_bar_rejection(df):
    long_wick = df["lower_wick"] > 2 * df["body"].abs()
    short_wick = df["upper_wick"] > 2 * df["body"].abs()
    up = long_wick & (df["lower_wick"] > 0.008) & (df["close"] > df["open"])
    dn = short_wick & (df["upper_wick"] > 0.008) & (df["close"] < df["open"])
    return _sig(up, dn, df.index)


def three_bar_reversal(df):
    c1, c2, c3 = df["close"].shift(2), df["close"].shift(1), df["close"]
    up = (c1 < df["open"].shift(2)) & (c2 < c1) & (c3 > df["open"]) & (c3 > c2)
    dn = (c1 > df["open"].shift(2)) & (c2 > c1) & (c3 < df["open"]) & (c3 < c2)
    return _sig(up, dn, df.index)


def nr7_breakout(df):
    """Range contraction then expansion - a Toby Crabel idea."""
    up = df["nr7"].shift(1).fillna(0).astype(bool) & (df["close"] > df["high"].shift(1))
    dn = df["nr7"].shift(1).fillna(0).astype(bool) & (df["close"] < df["low"].shift(1))
    return _sig(up, dn, df.index)


def williams_volatility_breakout(df):
    """Larry Williams: buy the break of the open plus a volatility unit."""
    unit = df["atr"].shift(1)
    up = df["close"] > df["open"] + unit
    dn = df["close"] < df["open"] - unit
    return _sig(up, dn, df.index)


def atr_expansion(df):
    quiet = df["atr_rank"] < 0.30
    up = quiet.shift(1).fillna(False) & (df["close"] > df["high"].shift(1)) & (
        df["tr"] > df["atr"] * 1.5
    )
    dn = quiet.shift(1).fillna(False) & (df["close"] < df["low"].shift(1)) & (
        df["tr"] > df["atr"] * 1.5
    )
    return _sig(up, dn, df.index)


def vwap_reclaim(df):
    up = _cross_up(df["close"], df["vwap20"]) & (df["close"] > df["ema200"])
    dn = _cross_dn(df["close"], df["vwap20"]) & (df["close"] < df["ema200"])
    return _sig(up, dn, df.index)


def volume_surge_breakout(df):
    up = (df["vol_ratio"] > 2) & (df["close"] > df["high"].shift(1)) & (df["close"] > df["open"])
    dn = (df["vol_ratio"] > 2) & (df["close"] < df["low"].shift(1)) & (df["close"] < df["open"])
    return _sig(up, dn, df.index)


def obv_divergence(df):
    up = (df["obv"] > df["obv_ma"]) & (df["obv_ma"] > df["obv_ma"].shift(1)) & (
        df["close"] > df["sma50"]
    )
    dn = (df["obv"] < df["obv_ma"]) & (df["obv_ma"] < df["obv_ma"].shift(1)) & (
        df["close"] < df["sma50"]
    )
    return _sig(up & ~up.shift(1).fillna(False), dn & ~dn.shift(1).fillna(False), df.index)


# ---------------------------------------------------------------------------
# 47-50  cross-sectional and long-horizon momentum
# ---------------------------------------------------------------------------
def tsmom(df):
    """Time-series momentum (Moskowitz, Ooi & Pedersen)."""
    return _sig(df["close"] > df["close"].shift(42), df["close"] < df["close"].shift(42), df.index)


def dual_momentum(df):
    """Gary Antonacci's dual momentum: absolute plus relative trend."""
    absolute_up = df["close"] > df["sma200"]
    absolute_dn = df["close"] < df["sma200"]
    rel_up = df["close"] > df["close"].shift(63)
    rel_dn = df["close"] < df["close"].shift(63)
    up = absolute_up & rel_up & ~(absolute_up & rel_up).shift(1).fillna(False)
    dn = absolute_dn & rel_dn & ~(absolute_dn & rel_dn).shift(1).fillna(False)
    return _sig(up, dn, df.index)


def breakout_retest(df):
    broke = (df["close"] > df["dc_high20"]) & (df["close"].shift(1) <= df["dc_high20"].shift(1))
    hold = broke.rolling(6).max().fillna(0).astype(bool)
    up = hold & (df["low"] <= df["dc_high20"]) & (df["close"] > df["dc_high20"])
    broke_dn = (df["close"] < df["dc_low20"]) & (df["close"].shift(1) >= df["dc_low20"].shift(1))
    hold_dn = broke_dn.rolling(6).max().fillna(0).astype(bool)
    dn = hold_dn & (df["high"] >= df["dc_low20"]) & (df["close"] < df["dc_low20"])
    return _sig(up, dn, df.index)


def trend_pullback_ema(df):
    """Trend plus pullback: the shape Amir originally asked for, on 4h."""
    up_trend = (df["ema20"] > df["ema50"]) & (df["close"] > df["ema200"])
    dn_trend = (df["ema20"] < df["ema50"]) & (df["close"] < df["ema200"])
    up = up_trend & (df["low"] <= df["ema20"]) & (df["close"] > df["ema20"]) & (df["rsi"] > 50)
    dn = dn_trend & (df["high"] >= df["ema20"]) & (df["close"] < df["ema20"]) & (df["rsi"] < 50)
    return _sig(up, dn, df.index)


STRATEGIES: dict[str, callable] = {
    "ema_cross_20_50": ema_cross_20_50,
    "ema_cross_9_21": ema_cross_9_21,
    "ema_cross_13_34": ema_cross_13_34,
    "guppy_mma": guppy_mma,
    "macd_cross": macd_cross,
    "macd_hist_flip": macd_hist_flip,
    "elder_triple_screen": elder_triple_screen,
    "elder_impulse": elder_impulse,
    "connors_rsi2": connors_rsi2,
    "connors_rsi2_exit_trend": connors_rsi2_exit_trend,
    "rsi14_pullback": rsi14_pullback,
    "rsi_divergence": rsi_divergence,
    "stoch_cross": stoch_cross,
    "stoch_reversal": stoch_reversal,
    "cci_zero_cross": cci_zero_cross,
    "cci_extreme_fade": cci_extreme_fade,
    "bollinger_breakout": bollinger_breakout,
    "bollinger_bounce": bollinger_bounce,
    "bollinger_pctb_reversal": bollinger_pctb_reversal,
    "keltner_breakout": keltner_breakout,
    "keltner_bounce": keltner_bounce,
    "zscore_reversion": zscore_reversion,
    "turtle_donchian20": turtle_donchian20,
    "turtle_donchian55": turtle_donchian55,
    "turtle_soup": turtle_soup,
    "darvas_box": darvas_box,
    "weinstein_stage2": weinstein_stage2,
    "minervini_trend_template": minervini_trend_template,
    "oneil_breakout": oneil_breakout,
    "livermore_pivot": livermore_pivot,
    "wyckoff_spring": wyckoff_spring,
    "market_structure": market_structure,
    "bos_retest": bos_retest,
    "adx_trend": adx_trend,
    "adx_rising_trend": adx_rising_trend,
    "supertrend_flip": supertrend_flip,
    "psar_flip": psar_flip,
    "ichimoku_cloud": ichimoku_cloud,
    "ichimoku_tk": ichimoku_tk,
    "heikin_ashi_trend": heikin_ashi_trend,
    "aroon_trend": aroon_trend,
    "slope_momentum": slope_momentum,
    "inside_bar_breakout": inside_bar_breakout,
    "engulfing_pattern": engulfing_pattern,
    "pin_bar_rejection": pin_bar_rejection,
    "three_bar_reversal": three_bar_reversal,
    "nr7_breakout": nr7_breakout,
    "williams_volatility_breakout": williams_volatility_breakout,
    "atr_expansion": atr_expansion,
    "vwap_reclaim": vwap_reclaim,
    "volume_surge_breakout": volume_surge_breakout,
    "obv_divergence": obv_divergence,
    "tsmom": tsmom,
    "dual_momentum": dual_momentum,
    "breakout_retest": breakout_retest,
    "trend_pullback_ema": trend_pullback_ema,
}
