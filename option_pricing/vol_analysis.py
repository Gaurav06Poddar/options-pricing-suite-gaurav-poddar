#option-pricing-models/option_pricing/vol_analysis.py
"""
Volatility analysis helpers: realized vs implied volatility, term structure, and comparisons.

This module provides:
- compute_rolling_realized_vs_implied(close_series, iv, window_days=21, freq_per_day=252)
- aggregate_implied_term_structure(ticker, expiries=None, r=0.01)
- choose_atm_iv_from_chain(df_iv, spot)
- fetch_expiries(ticker)
- compute_realized_vs_implied_surface(ticker, expiries, window_days=21, r=0.0)

Notes:
- Many helpers use yfinance under the hood (via market_iv_surface). Network IO may raise exceptions
  if Yahoo's service is unavailable; callers should handle exceptions or run in environments with network access.
- Realized volatility functions expect pandas Series indexed by date with Close prices.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from .realized_vol import rolling_realized_volatility
from .market_iv_surface import implied_vol_from_chain, fetch_expiries, build_surface

# Local import placed here to avoid circular import if used in other modules at import-time
import yfinance as yf
import datetime


def compute_rolling_realized_vs_implied(close_series: pd.Series,
                                       iv: float,
                                       window_days: int = 21,
                                       freq_per_day: int = 252) -> pd.DataFrame:
    """
    Compute rolling realized volatility (annualized) from close prices and compare to a single implied vol.

    Returns a DataFrame indexed like the input close_series (right-aligned for the rolling window)
    with columns: ['realized_vol', 'implied_vol', 'rv_to_iv_ratio'].

    Arguments:
      close_series: pandas Series of close prices indexed by date
      iv: implied volatility (annualized, decimal, e.g. 0.2)
      window_days: rolling window in trading days for realized vol (default 21)
      freq_per_day: trading days per year (default 252)
    """
    if not isinstance(close_series, pd.Series):
        raise ValueError("close_series must be a pandas Series of close prices indexed by date")

    rv_series = rolling_realized_volatility(close_series, window=window_days, freq_per_day=freq_per_day)
    df = pd.DataFrame({'realized_vol': rv_series})
    df['implied_vol'] = float(iv) if iv is not None else np.nan
    # avoid division by zero
    df['rv_to_iv_ratio'] = df['realized_vol'] / (df['implied_vol'] + 1e-12)
    return df.dropna()


def choose_atm_iv_from_chain(df_iv: pd.DataFrame, spot: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Given an implied-vol DataFrame (as returned by implied_vol_from_chain) indexed by strike,
    choose the ATM strike closest to spot and return (atm_strike, atm_iv) where atm_iv is the call_iv if
    available, otherwise the put_iv. Returns (None, None) if df_iv is empty or invalid.

    df_iv expected to have index = strikes and columns that include 'call_iv' and 'put_iv' (may be NaN).
    """
    if df_iv is None or df_iv.empty:
        return None, None
    strikes = np.asarray(df_iv.index.values, dtype=float)
    if strikes.size == 0:
        return None, None
    atm_idx = int(np.argmin(np.abs(strikes - float(spot))))
    atm_strike = strikes[atm_idx]
    row = df_iv.loc[atm_strike]
    call_iv = row.get('call_iv', np.nan)
    put_iv = row.get('put_iv', np.nan)
    if not pd.isna(call_iv):
        return float(atm_strike), float(call_iv)
    if not pd.isna(put_iv):
        return float(atm_strike), float(put_iv)
    return float(atm_strike), None


def aggregate_implied_term_structure(ticker: str,
                                     expiries: Optional[List[str]] = None,
                                     r: float = 0.01) -> pd.DataFrame:
    """
    For a ticker, fetch option chains and compute an ATM implied vol per expiry.

    Returns DataFrame with columns:
      ['expiry', 'days_to_expiry', 'atm_strike', 'atm_iv', 'spot']

    If expiries is None, the function will iterate all expiries available from yfinance.
    """
    try:
        chains = implied_vol_from_chain(ticker)
    except Exception:
        # Best-effort: try using yfinance directly for expiries list
        try:
            t = yf.Ticker(ticker)
            exps = list(t.options or [])
            chains = {}
            for e in exps:
                try:
                    c = t.option_chain(e)
                    chains[e] = {'calls': c.calls, 'puts': c.puts}
                except Exception:
                    continue
        except Exception:
            return pd.DataFrame()

    if not chains:
        return pd.DataFrame()

    exps = sorted(list(chains.keys()))
    if expiries is not None:
        exps = [e for e in exps if e in expiries]

    rows: List[Dict[str, Any]] = []
    t_yf = yf.Ticker(ticker)
    # get spot (most recent close) once; fallback if missing
    spot = None
    try:
        hist = t_yf.history(period='1d')
        if hist is not None and not hist.empty:
            spot = float(hist['Close'].iloc[-1])
    except Exception:
        spot = None

    today = datetime.date.today()

    for e in exps:
        try:
            result = implied_vol_from_chain(ticker, e, spot=spot, r=r)
            if result is None:
                continue
            else:
                df_iv, underlying_price, rfr = result
                if df_iv.empty:
                    continue

            # get spot if not set earlier
            if spot is None:
                try:
                    hist = t_yf.history(period='1d')
                    spot = float(hist['Close'].iloc[-1])
                except Exception:
                    spot = None

            expiry_date = datetime.datetime.strptime(e, "%Y-%m-%d").date()
            days = max((expiry_date - today).days, 1)

            strikes = np.asarray(df_iv.index.values, dtype=float)
            if strikes.size == 0:
                continue
            atm_idx = int(np.argmin(np.abs(strikes - (spot if spot is not None else strikes.mean()))))
            atm_strike = strikes[atm_idx]
            row = df_iv.loc[atm_strike]
            call_iv = row.get('call_iv', np.nan)
            put_iv = row.get('put_iv', np.nan)
            atm_iv = None
            if not pd.isna(call_iv):
                atm_iv = float(call_iv)
            elif not pd.isna(put_iv):
                atm_iv = float(put_iv)
            else:
                atm_iv = None

            rows.append({
                'expiry': e,
                'days_to_expiry': int(days),
                'atm_strike': float(atm_strike),
                'atm_iv': atm_iv,
                'spot': float(spot) if spot is not None else None
            })
        except Exception:
            # skip problematic expiries – be conservative
            continue

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).sort_values('days_to_expiry')
    # normalize types and handle missing IV elegantly
    df_out['atm_iv'] = df_out['atm_iv'].astype(float, errors='ignore')
    return df_out.reset_index(drop=True)


def fetch_expiries(ticker: str) -> List[str]:
    """
    Helper: return sorted list of expiry strings (YYYY-MM-DD) for a ticker using yfinance.
    Returns empty list on failure.
    """
    try:
        t = yf.Ticker(ticker)
        opts = list(t.options or [])
        return sorted(opts)
    except Exception:
        return []


def compute_realized_vs_implied_surface(ticker: str,
                                        expiries: List[str],
                                        window_days: int = 21,
                                        r: float = 0.0) -> pd.DataFrame:
    """
    For each expiry in expiries (list of 'YYYY-MM-DD'), compute:
      - ATM implied vol (from implied_vol_from_chain)
      - realized vol over the recent window_days from historical closes

    Returns DataFrame with columns:
      ['expiry', 'days_to_expiry', 'atm_strike', 'atm_iv', 'realized_vol', 'spot']

    Behavior:
      - If historical data or IV cannot be fetched for a given expiry, that expiry is skipped.
      - realized_vol is the latest available rolling realized volatility (annualized) over window_days.
    """
    t = yf.Ticker(ticker)
    # fetch historical closes (1 year)
    try:
        hist = t.history(period='365d')
    except Exception:
        hist = None

    if hist is None or hist.empty:
        # can't compute realized vol without history; still attempt to return implied vols only with NaN realized
        use_realized = False
    else:
        use_realized = True
        close = hist['Close'].dropna()
        rv_series = rolling_realized_volatility(close, window=window_days, freq_per_day=252)
        latest_rv = float(rv_series.dropna().iloc[-1]) if (rv_series is not None and rv_series.dropna().size > 0) else np.nan

    rows: List[Dict[str, Any]] = []
    today = datetime.date.today()
    # get spot once if possible
    spot = None
    try:
        spot = float(t.history(period='1d')['Close'].iloc[-1])
    except Exception:
        spot = None

    for e in expiries:
        try:
            df_iv = implied_vol_from_chain(ticker, e, spot=spot, r=r)
            if df_iv is None or df_iv.empty:
                continue
            # update spot if missing
            if spot is None:
                try:
                    spot = float(t.history(period='1d')['Close'].iloc[-1])
                except Exception:
                    spot = None

            strikes = np.asarray(df_iv.index.values, dtype=float)
            if strikes.size == 0:
                continue
            atm_idx = int(np.argmin(np.abs(strikes - (spot if spot is not None else strikes.mean()))))
            atm_strike = strikes[atm_idx]
            row = df_iv.loc[atm_strike]
            call_iv = row.get('call_iv', np.nan)
            put_iv = row.get('put_iv', np.nan)
            atm_iv = None
            if not pd.isna(call_iv):
                atm_iv = float(call_iv)
            elif not pd.isna(put_iv):
                atm_iv = float(put_iv)
            else:
                atm_iv = np.nan

            expiry_date = datetime.datetime.strptime(e, "%Y-%m-%d").date()
            days = max((expiry_date - today).days, 1)

            rows.append({
                'expiry': e,
                'days_to_expiry': int(days),
                'atm_strike': float(atm_strike),
                'atm_iv': atm_iv,
                'realized_vol': float(latest_rv) if use_realized and not np.isnan(latest_rv) else np.nan,
                'spot': float(spot) if spot is not None else None
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df_res = pd.DataFrame(rows).sort_values('days_to_expiry').reset_index(drop=True)
    return df_res
