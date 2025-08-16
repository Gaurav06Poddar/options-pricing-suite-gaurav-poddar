# option-pricing/market_iv_surface.py
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from functools import lru_cache

from .BlackScholesModel import BlackScholesModel

def _days_to(expiry_str, today=None):
    if today is None:
        today = dt.date.today()
    e = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
    d = (e - today).days
    return max(d, 1)

@lru_cache(maxsize=64)
def fetch_expiries(ticker: str):
    try:
        return tuple(yf.Ticker(ticker).options)
    except Exception:
        return tuple()

@lru_cache(maxsize=32)
def fetch_chain(ticker: str, expiry: str):
    t = yf.Ticker(ticker)
    ch = t.option_chain(expiry)
    return ch.calls.copy(), ch.puts.copy()

def implied_vol_from_chain(ticker: str, expiry: str, spot: float=None, r=0.01):
    """
    Compute implied vol for each strike in the selected expiry's option chain.
    Returns DataFrame with: strike, call_mid, put_mid, call_iv, put_iv
    """
    t = yf.Ticker(ticker)
    if spot is None:
        spot = float(t.history(period='1d')['Close'].iloc[-1])
    try:
        calls, puts = fetch_chain(ticker, expiry)
    except Exception:
        # fallback direct fetch
        ch = t.option_chain(expiry)
        calls, puts = ch.calls.copy(), ch.puts.copy()

    def mid_price(row):
        b = row.get('bid', np.nan); a = row.get('ask', np.nan)
        if not (np.isnan(b) or np.isnan(a)) and a > 0:
            return 0.5 * (b + a)
        for k in ['lastPrice', 'last', 'mid']:
            v = row.get(k, np.nan)
            if v is not None and not np.isnan(v) and v > 0:
                return v
        return np.nan

    strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
    days = _days_to(expiry)

    rows = []
    for K in strikes:
        cr = calls[calls['strike'] == K]
        pr = puts[puts['strike'] == K]
        call_mid = mid_price(cr.iloc[0].to_dict()) if len(cr) else np.nan
        put_mid = mid_price(pr.iloc[0].to_dict()) if len(pr) else np.nan

        call_iv = np.nan
        put_iv = np.nan
        if np.isfinite(call_mid) and call_mid > 1e-10:
            bsm = BlackScholesModel(spot, K, days, r, 0.2)
            try:
                call_iv = bsm.implied_volatility(call_mid, option_type='call')
            except Exception:
                call_iv = np.nan
        if np.isfinite(put_mid) and put_mid > 1e-10:
            bsm = BlackScholesModel(spot, K, days, r, 0.2)
            try:
                put_iv = bsm.implied_volatility(put_mid, option_type='put')
            except Exception:
                put_iv = np.nan

        rows.append({
            'strike': K,
            'call_mid': call_mid,
            'put_mid': put_mid,
            'call_iv': call_iv,
            'put_iv': put_iv,
        })

    df = pd.DataFrame(rows).sort_values('strike').reset_index(drop=True)
    # make strikes the index for convenient lookup elsewhere in code
    if 'strike' in df.columns:
        df.set_index('strike', inplace=True)
    # also return spot and T for convenience
    T = days / 365.0
    return df, spot, T


def build_surface(ticker: str, expiries: list[str], r=0.01, use_calls=True):
    """
    Returns long-form DataFrame with columns: expiry, T, strike, iv
    """
    t = yf.Ticker(ticker)
    spot = float(t.history(period='1d')['Close'].iloc[-1])
    rows = []
    for e in expiries:
        df, _, T = implied_vol_from_chain(ticker, e, spot=spot, r=r)
        col = 'call_iv' if use_calls else 'put_iv'
        for _, row in df.iterrows():
            iv = row[col]
            if iv is not None and np.isfinite(iv):
                rows.append({'expiry': e, 'T': T, 'strike': row['strike'], 'iv': float(iv)})
    return pd.DataFrame(rows)
