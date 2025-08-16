# option-pricing/realized_vol.py
"""
Realized volatility estimators and related stats.

Functions:
- log_returns(prices)
- realized_variance(returns, freq_per_day)
- rolling_realized_volatility(series, window=21, freq_per_day=252)
- bipower_variation(returns)
- rolling_bipower(series, window)
- annualize(vol, periods_per_year)
"""

import numpy as np
import pandas as pd

def log_returns(prices: pd.Series) -> pd.Series:
    """
    Natural log returns from price series (index datetime).
    """
    prices = prices.astype(float)
    return np.log(prices).diff().dropna()

def realized_variance(returns: np.ndarray) -> float:
    """
    Returns realized variance (sum of squared returns). Input returns assumed to be returns
    for the period (e.g. per-minute log returns). Does not annualize.
    """
    r = np.asarray(returns, dtype=float)
    return np.sum(r**2)

def realized_volatility_from_returns(returns: np.ndarray, periods_per_year: int) -> float:
    """
    Annualized realized volatility from returns array.
    """
    rv = realized_variance(returns)
    # realized variance per observation -> annualize multiplying by periods_per_year
    return np.sqrt(rv * periods_per_year)

def rolling_realized_volatility(price_series: pd.Series, window: int = 21, freq_per_day: int = 252) -> pd.Series:
    """
    Rolling realized volatility based on daily prices.
    window: number of observations in each rolling bucket (days).
    freq_per_day: number of trading periods per year to annualize (252 typical).
    Returns series indexed by the right side (i.e., same index as the last day in window).
    """
    ret = log_returns(price_series)
    # Use sum of squared returns in window
    def rv_window(x):
        arr = np.asarray(x)
        return np.sqrt(np.sum(arr**2) * freq_per_day)

    # ret.rolling uses center default False -> right-aligned
    rv = ret.rolling(window=window).apply(lambda x: np.sqrt(np.nansum(x**2) * freq_per_day), raw=True)
    rv.name = "realized_vol"
    return rv

def bipower_variation(returns: np.ndarray) -> float:
    """
    Bipower variation estimator (Barndorff-Nielsen & Shephard).
    For returns r_i, BV = (pi/2) * sum_{i=2..n} |r_i| * |r_{i-1}|
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    return (np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))

def rolling_bipower(price_series: pd.Series, window: int = 21, freq_per_day: int = 252) -> pd.Series:
    """
    Rolling bipower variation converted to annualized volatility proxy.
    """
    ret = log_returns(price_series)
    def bv_window(x):
        arr = np.asarray(x)
        if arr.size < 2:
            return np.nan
        bv = (np.pi / 2.0) * np.sum(np.abs(arr[1:]) * np.abs(arr[:-1]))
        return np.sqrt(bv * freq_per_day)
    bv = ret.rolling(window=window).apply(bv_window, raw=True)
    bv.name = "bipower_vol"
    return bv

def annualize_vol(vol, scale_factor=1.0):
    """
    If vol calculated for daily (or already annualized) you can map using scale_factor.
    Typically not needed if functions above already annualize.
    """
    return vol * scale_factor
