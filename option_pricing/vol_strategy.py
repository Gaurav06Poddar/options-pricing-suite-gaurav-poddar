#option-pricing-models/option_pricing/vol_strategy.py
"""
Vol-arbitrage backtest and grid-search utilities.

This module performs a simple volatility-arbitrage strategy: short an ATM straddle when implied
volatility is "rich" relative to realized volatility, and delta-hedge the position. It uses the
hedging.backtest_dynamic_hedge routine to execute the dynamic hedging backtest for each leg.

Features:
- Robust ATM implied vol extraction from yfinance option chains (via implied_vol_from_chain)
- Realized vol estimation fallback (rolling realized vol)
- Execution cost forwarding (tc_per_share, impact_per_share, impact_coeff_quadratic, liquidity_scale)
- Grid search across IV/RV thresholds and rebalance frequencies to evaluate strategy performance
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List

from .hedging import simulate_black_scholes_paths, backtest_dynamic_hedge
from .market_iv_surface import implied_vol_from_chain
from .realized_vol import rolling_realized_volatility

import yfinance as yf
import datetime


def vol_arbitrage_strategy_backtest(ticker: str,
                                   expiry: str,
                                   n_paths: int = 2000,
                                   n_steps: int = 50,
                                   threshold: float = 1.05,
                                   use_realized_sigma: bool = True,
                                   sigma_override: Optional[float] = None,
                                   rebalance_every: int = 1,
                                   tc_per_share: float = 0.0,
                                   impact_per_share: float = 0.0,
                                   impact_coeff_quadratic: float = 0.0,
                                   liquidity_scale: float = 1e6,
                                   r: float = 0.0) -> Dict:
    """
    Performs vol-arb backtest for an ATM short straddle with delta hedging.

    Args:
      ticker: ticker symbol understood by yfinance
      expiry: expiry string in 'YYYY-MM-DD' format (must be present in yfinance option expiries)
      n_paths, n_steps: simulation parameters for path generation
      threshold: trade when ATM_iv / realized_sigma > threshold
      use_realized_sigma: whether to estimate realized sigma from recent historical data
      sigma_override: if provided and use_realized_sigma is False, uses this sigma for simulating true paths
      rebalance_every: hedging rebalance frequency in steps
      tc_per_share: fixed per-share transaction cost
      impact_per_share: linear slippage per share ($)
      impact_coeff_quadratic: quadratic impact coefficient (dimensionless)
      liquidity_scale: notional scale for quadratic impact normalization
      r: risk-free rate (annual decimal)

    Returns:
      results dict containing metadata, trading decision, and if traded:
        - pnl_straddle: numpy array of per-path discounted P&L (short call + short put hedged)
        - summary: mean/std/median/quantiles of pnl_straddle
        - details_call/details_put: details from hedging backtests
    """
    # fetch IV chain and ATM iv
    result = implied_vol_from_chain(ticker, expiry, spot=None, r=r)
    if result is None:
        raise RuntimeError("No implied vol data found for selected expiry.")
    else:
        df_iv, underlying_price, rfr = result
        if df_iv.empty:
            raise RuntimeError("No implied vol data found for selected expiry.")
    t = yf.Ticker(ticker)
    try:
        spot = float(t.history(period='1d')['Close'].iloc[-1])
    except Exception as e:
        raise RuntimeError(f"Failed to fetch current spot for {ticker}: {e}")

    strikes = np.array(df_iv.index.values, dtype=float)
    if strikes.size == 0:
        raise RuntimeError("No strikes available in option chain.")

    atm_idx = int(np.argmin(np.abs(strikes - spot)))
    atm_strike = strikes[atm_idx]
    atm_row = df_iv.loc[atm_strike]
    # atm_iv = atm_row['call_iv'] if not pd.isna(atm_row.get('call_iv', np.nan)) else atm_row.get('put_iv', np.nan)
    # if atm_iv is None or np.isnan(atm_iv):
    #     raise RuntimeError("ATM IV unavailable.")
    atm_iv = atm_row.get('call_iv', np.nan)
    if np.isnan(atm_iv):
        atm_iv = atm_row.get('put_iv', np.nan)
    if np.isnan(atm_iv):
        # try average if both missing
        atm_iv = np.nanmean([atm_row.get('call_iv', np.nan), atm_row.get('put_iv', np.nan)])
    if np.isnan(atm_iv):
        raise RuntimeError(f"ATM IV unavailable at strike {atm_strike}. Row: {atm_row.to_dict()}")

    # realized sigma estimation (annualized)
    if use_realized_sigma:
        hist = t.history(period='180d')
        if hist is None or hist.empty:
            realized_sigma = atm_iv  # fallback
        else:
            close = hist['Close'].dropna()
            rv_series = rolling_realized_volatility(close, window=21, freq_per_day=252)
            realized_sigma = float(rv_series.dropna().iloc[-1]) if rv_series.dropna().size else atm_iv
    else:
        realized_sigma = sigma_override if sigma_override is not None else atm_iv

    trade_signal = (float(atm_iv) / (realized_sigma + 1e-12)) > threshold

    # compute T (years) from expiry
    try:
        expiry_date = datetime.datetime.strptime(expiry, "%Y-%m-%d").date()
        today = datetime.date.today()
        days = max((expiry_date - today).days, 1)
        T = days / 365.0
    except Exception:
        T = 30.0 / 365.0

    # simulate GBM paths using realized_sigma as true sigma for underlying dynamics
    paths = simulate_black_scholes_paths(spot, r, realized_sigma, T, n_paths=n_paths, n_steps=n_steps, seed=42)

    results = {
        'ticker': ticker,
        'expiry': expiry,
        'atm_strike': float(atm_strike),
        'atm_iv': float(atm_iv),
        'realized_sigma': float(realized_sigma),
        'sigma_true': float(realized_sigma),
        'trade_signal': bool(trade_signal),
        'T': float(T),
        'n_paths': n_paths,
        'n_steps': n_steps,
    }

    if not trade_signal:
        results['message'] = f"No trade: ATM IV ({atm_iv:.4f}) not greater than realized*threshold ({realized_sigma*threshold:.4f})."
        return results

    # backtest both legs; pass execution cost parameters to the hedging backtester
    df_call, summary_call, details_call = backtest_dynamic_hedge(
        paths, strike=atm_strike, r=r, sigma_model=float(atm_iv), T=T, option_type='call',
        rebalance_every=rebalance_every, tc_per_share=tc_per_share,
        impact_per_share=impact_per_share, impact_coeff_quadratic=impact_coeff_quadratic, liquidity_scale=liquidity_scale
    )

    df_put, summary_put, details_put = backtest_dynamic_hedge(
        paths, strike=atm_strike, r=r, sigma_model=float(atm_iv), T=T, option_type='put',
        rebalance_every=rebalance_every, tc_per_share=tc_per_share,
        impact_per_share=impact_per_share, impact_coeff_quadratic=impact_coeff_quadratic, liquidity_scale=liquidity_scale
    )

    # Each df has 'pnl_disc' column; short call + short put -> pnl = pnl_call + pnl_put
    pnl_call = df_call['pnl_disc'].values
    pnl_put = df_put['pnl_disc'].values
    pnl_straddle = pnl_call + pnl_put

    results['pnl_straddle'] = pnl_straddle
    results['summary'] = {
        'mean_pnl': float(np.mean(pnl_straddle)),
        'std_pnl': float(np.std(pnl_straddle, ddof=1)),
        'median_pnl': float(np.median(pnl_straddle)),
        'pnl_5pct': float(np.quantile(pnl_straddle, 0.05)),
        'pnl_95pct': float(np.quantile(pnl_straddle, 0.95)),
    }
    results['details_call'] = details_call
    results['details_put'] = details_put
    results['summary_call'] = summary_call
    results['summary_put'] = summary_put

    return results


def grid_search_vol_arb(ticker: str,
                        expiry: str,
                        thresholds: List[float],
                        rebalance_list: List[int],
                        n_paths: int = 2000,
                        n_steps: int = 50,
                        tc_per_share: float = 0.0,
                        impact_per_share: float = 0.0,
                        impact_coeff_quadratic: float = 0.0,
                        liquidity_scale: float = 1e6,
                        r: float = 0.0,
                        max_cells: int = 100) -> Dict:
    """
    Run grid search of vol-arb backtest across arrays of thresholds x rebalance frequencies.

    Returns dict:
      {
        'thresholds': thresholds,
        'rebalance_list': rebalance_list,
        'mean_matrix': 2D np.array (len(rebalance_list), len(thresholds)),
        'std_matrix': 2D np.array,
        'sharpe_matrix': 2D np.array (mean/std),
        'raw': mapping for each cell with result dict or error
      }

    Notes:
      - For speed, keep n_paths small when scanning larger grids.
      - max_cells guard prevents runaway computation.
    """
    n_thr = len(thresholds)
    n_reb = len(rebalance_list)
    total_cells = n_thr * n_reb
    if total_cells > max_cells:
        raise ValueError(f"Grid too large ({total_cells} cells). Reduce grid or increase max_cells.")

    mean_mat = np.full((n_reb, n_thr), np.nan, dtype=float)
    std_mat = np.full((n_reb, n_thr), np.nan, dtype=float)
    sharpe_mat = np.full((n_reb, n_thr), np.nan, dtype=float)
    raw = {}

    for i, reb in enumerate(rebalance_list):
        for j, thr in enumerate(thresholds):
            key = f"r{reb}_t{thr}"
            try:
                res = vol_arbitrage_strategy_backtest(
                    ticker=ticker,
                    expiry=expiry,
                    n_paths=n_paths,
                    n_steps=n_steps,
                    threshold=thr,
                    rebalance_every=reb,
                    tc_per_share=tc_per_share,
                    impact_per_share=impact_per_share,
                    impact_coeff_quadratic=impact_coeff_quadratic,
                    liquidity_scale=liquidity_scale,
                    r=r
                )
                if not res.get('trade_signal', False):
                    # no trade signaled; record zeros (or could set NaN)
                    mean = 0.0
                    std = 0.0
                else:
                    pnl = res['pnl_straddle']
                    mean = float(np.mean(pnl))
                    std = float(np.std(pnl, ddof=1))
                mean_mat[i, j] = mean
                std_mat[i, j] = std
                sharpe_mat[i, j] = mean / std if (std is not None and std > 0.0) else np.nan
                raw[key] = res
            except Exception as e:
                mean_mat[i, j] = np.nan
                std_mat[i, j] = np.nan
                sharpe_mat[i, j] = np.nan
                raw[key] = {'error': str(e)}

    return {
        'thresholds': thresholds,
        'rebalance_list': rebalance_list,
        'mean_matrix': mean_mat,
        'std_matrix': std_mat,
        'sharpe_matrix': sharpe_mat,
        'raw': raw
    }
