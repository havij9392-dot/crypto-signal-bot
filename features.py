"""
Feature engineering for the strategy sweep.

Every indicator here is computed from closed bars only. Nothing is shifted
backwards in time, so a signal at bar i can be acted on at the open of bar i+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing - the average used by RSI, ATR and ADX."""
    return series.ewm(alpha=1 / length, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # --- moving averages ----------------------------------------------------
    for span in (9, 13, 20, 21, 34, 50, 100, 200):
        df[f"ema{span}"] = _ema(close, span)
    for span in (10, 20, 50, 200):
        df[f"sma{span}"] = close.rolling(span).mean()

    # --- range / volatility -------------------------------------------------
    df["tr"] = _true_range(df)
    df["atr"] = _rma(df["tr"], 14)
    df["atr_pct"] = df["atr"] / close
    df["atr_rank"] = df["atr_pct"].rolling(100).rank(pct=True)
    df["vol_sma20"] = volume.rolling(20).mean()

    # --- RSI ----------------------------------------------------------------
    delta = close.diff()
    gain = _rma(delta.clip(lower=0), 14)
    loss = _rma((-delta).clip(lower=0), 14)
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["rsi2"] = (
        100
        - 100
        / (
            1
            + _rma(delta.clip(lower=0), 2) / _rma((-delta).clip(lower=0), 2).replace(0, np.nan)
        )
    )

    # --- directional movement / ADX ----------------------------------------
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    df["plus_di"] = 100 * _rma(pd.Series(plus_dm, index=df.index), 14) / df["atr"].replace(0, np.nan)
    df["minus_di"] = 100 * _rma(pd.Series(minus_dm, index=df.index), 14) / df["atr"].replace(0, np.nan)
    dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (
        (df["plus_di"] + df["minus_di"]).replace(0, np.nan)
    )
    df["adx"] = _rma(dx, 14)

    # --- MACD ---------------------------------------------------------------
    macd = _ema(close, 12) - _ema(close, 26)
    df["macd"] = macd
    df["macd_signal"] = _ema(macd, 9)
    df["macd_hist"] = macd - df["macd_signal"]

    # --- stochastic ---------------------------------------------------------
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- Bollinger ----------------------------------------------------------
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    df["bb_mid"] = mid
    df["bb_up"] = mid + 2 * sd
    df["bb_dn"] = mid - 2 * sd
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / mid.replace(0, np.nan)
    df["bb_pctb"] = (close - df["bb_dn"]) / (df["bb_up"] - df["bb_dn"]).replace(0, np.nan)
    df["bb_width_rank"] = df["bb_width"].rolling(100).rank(pct=True)

    # --- Keltner ------------------------------------------------------------
    df["kc_up"] = _ema(close, 20) + 2 * df["atr"]
    df["kc_dn"] = _ema(close, 20) - 2 * df["atr"]

    # --- Donchian -----------------------------------------------------------
    for span in (10, 20, 55):
        df[f"dc_high{span}"] = high.rolling(span).max().shift(1)
        df[f"dc_low{span}"] = low.rolling(span).min().shift(1)

    # --- CCI ----------------------------------------------------------------
    tp = (high + low + close) / 3
    ma = tp.rolling(20).mean()
    md = (tp - ma).abs().rolling(20).mean()
    df["cci"] = (tp - ma) / (0.015 * md.replace(0, np.nan))

    # --- Aroon --------------------------------------------------------------
    df["aroon_up"] = (
        high.rolling(25).apply(lambda w: float(np.argmax(w)) + 1, raw=True) / 25 * 100
    )
    df["aroon_dn"] = (
        low.rolling(25).apply(lambda w: float(np.argmin(w)) + 1, raw=True) / 25 * 100
    )

    # --- Ichimoku -----------------------------------------------------------
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df["tenkan"] = tenkan
    df["kijun"] = kijun
    df["senkou_a"] = ((tenkan + kijun) / 2).shift(26)
    df["senkou_b"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    # --- SuperTrend ---------------------------------------------------------
    # The bands have to ratchet: the lower band may only rise while the trend is
    # up, otherwise price never falls through it and the indicator never flips.
    hl2 = (high + low) / 2
    basic_up = (hl2 + 3 * df["atr"]).to_numpy(float)
    basic_dn = (hl2 - 3 * df["atr"]).to_numpy(float)
    c_arr = close.to_numpy(float)

    n_bars = len(df)
    final_up = np.full(n_bars, np.nan)
    final_dn = np.full(n_bars, np.nan)
    trend = np.ones(n_bars)

    for i in range(1, n_bars):
        if np.isnan(basic_up[i]) or np.isnan(basic_dn[i]):
            continue

        if np.isnan(final_up[i - 1]):
            final_up[i], final_dn[i] = basic_up[i], basic_dn[i]
            continue

        final_up[i] = (
            min(basic_up[i], final_up[i - 1])
            if c_arr[i - 1] <= final_up[i - 1]
            else basic_up[i]
        )
        final_dn[i] = (
            max(basic_dn[i], final_dn[i - 1])
            if c_arr[i - 1] >= final_dn[i - 1]
            else basic_dn[i]
        )

        if c_arr[i] > final_up[i - 1]:
            trend[i] = 1
        elif c_arr[i] < final_dn[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    df["st_dir"] = trend
    df["supertrend"] = np.where(trend > 0, final_dn, final_up)

    # --- Parabolic SAR ------------------------------------------------------
    psar = np.full(len(df), np.nan)
    h, lo = high.to_numpy(float), low.to_numpy(float)
    af, step, max_af = 0.02, 0.02, 0.2
    bullish, ep = True, h[0]
    psar[0] = lo[0]
    for i in range(1, len(df)):
        prev = psar[i - 1]
        if np.isnan(prev):
            psar[i] = lo[i - 1]
            continue
        candidate = prev + af * (ep - prev)
        if bullish:
            if lo[i] < candidate:
                bullish, candidate, ep, af = False, ep, lo[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_af)
        else:
            if h[i] > candidate:
                bullish, candidate, ep, af = True, ep, h[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, max_af)
        psar[i] = candidate
    df["psar"] = psar
    df["psar_bull"] = (close.to_numpy(float) > psar).astype(int)

    # --- volume -------------------------------------------------------------
    df["vol_ma20"] = volume.rolling(20).mean()
    df["vol_ratio"] = volume / df["vol_ma20"].replace(0, np.nan)
    df["obv"] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    df["obv_ma"] = df["obv"].rolling(20).mean()

    # --- price structure ----------------------------------------------------
    df["vwap20"] = (tp * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    df["zscore20"] = (close - df["sma20"]) / close.rolling(20).std().replace(0, np.nan)

    # linear-regression slope, normalised by price
    window = 20
    x = np.arange(window)
    x_centered = x - x.mean()
    denom = (x_centered**2).sum()

    def slope(values: np.ndarray) -> float:
        return float((x_centered * (values - values.mean())).sum() / denom)

    df["slope20"] = close.rolling(window).apply(slope, raw=True) / close

    # --- range contraction (NR4 / NR7) -------------------------------------
    rng = high - low
    df["nr4"] = (rng == rng.rolling(4).min()).astype(int)
    df["nr7"] = (rng == rng.rolling(7).min()).astype(int)

    # --- candles ------------------------------------------------------------
    df["body"] = (close - df["open"]) / df["open"]
    df["upper_wick"] = (high - np.maximum(close, df["open"])) / df["open"]
    df["lower_wick"] = (np.minimum(close, df["open"]) - low) / df["open"]

    return df


def swing_points(df: pd.DataFrame, k: int = 3) -> tuple[pd.Series, pd.Series]:
    """Confirmed pivot highs and lows, stamped at the bar that confirms them."""
    high, low = df["high"], df["low"]
    n = len(df)
    ph = pd.Series(np.nan, index=df.index)
    pl = pd.Series(np.nan, index=df.index)

    for i in range(k, n - k):
        window_h = high.iloc[i - k : i + k + 1]
        window_l = low.iloc[i - k : i + k + 1]
        if high.iloc[i] == window_h.max():
            ph.iloc[i + k] = high.iloc[i]
        if low.iloc[i] == window_l.min():
            pl.iloc[i + k] = low.iloc[i]

    return ph.ffill(), pl.ffill()


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha["open"] = (df["open"].shift(1) + df["close"].shift(1)) / 2
    ha["open"] = (ha["open"] + ha["close"].shift(1)) / 2
    ha["high"] = pd.concat([df["high"], ha["open"], ha["close"]], axis=1).max(axis=1)
    ha["low"] = pd.concat([df["low"], ha["open"], ha["close"]], axis=1).min(axis=1)
    return ha
