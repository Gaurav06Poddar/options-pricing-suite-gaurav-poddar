# option-pricing/microstructure.py
"""
Simple microstructure metrics using minute-level or tick-level price & volume.

Functions:
- trade_signs_tick_rule(prices): returns +1/-1 based on tick rule
- signed_volume(series_prices, volumes): returns signed volume series
- rolling_signed_volume_imbalance(signed_vol, window)
- vwap(series_prices, volumes, window=None)
"""

import numpy as np
import pandas as pd

def trade_signs_tick_rule(price_series: pd.Series) -> pd.Series:
    """
    Estimate trade sign by tick rule:
      sign = sign( price_t - price_{t-1} ), with ties using previous sign
    Returns series of +1/-1 with same index as price_series (first value NaN).
    """
    p = price_series.astype(float)
    diff = p.diff()
    signs = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    # Replace zeros by forward-fill of last non-zero sign (or 0 remains)
    signs = signs.replace(0, np.nan).ffill().fillna(0).astype(int)
    signs.name = "trade_sign"
    return signs

def signed_volume(price_series: pd.Series, volume_series: pd.Series) -> pd.Series:
    """
    Signed volume proxy: trade_signs * volume
    """
    s = trade_signs_tick_rule(price_series)
    signed = s * volume_series.astype(float)
    signed.name = "signed_volume"
    return signed

def rolling_signed_volume_imbalance(price_series: pd.Series, volume_series: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling imbalance: sum(signed_volume)/sum(abs(volume)) in the window.
    window is in number of observations (e.g., minutes).
    """
    sv = signed_volume(price_series, volume_series)
    abs_v = np.abs(volume_series.astype(float))
    num = sv.rolling(window=window).sum()
    den = abs_v.rolling(window=window).sum()
    imbalance = num / (den.replace(0, np.nan))
    imbalance.name = "signed_volume_imbalance"
    return imbalance

def vwap(price_series: pd.Series, volume_series: pd.Series, window: int = None) -> pd.Series:
    """
    VWAP series: rolling VWAP if window is provided, otherwise cumulative VWAP.
    """
    p = price_series.astype(float)
    v = volume_series.astype(float)
    pv = p * v
    if window is None:
        cum_pv = pv.cumsum()
        cum_v = v.cumsum().replace(0, np.nan)
        return (cum_pv / cum_v).rename("vwap")
    else:
        rp = pv.rolling(window=window).sum()
        rv = v.rolling(window=window).sum().replace(0, np.nan)
        return (rp / rv).rename("vwap")
