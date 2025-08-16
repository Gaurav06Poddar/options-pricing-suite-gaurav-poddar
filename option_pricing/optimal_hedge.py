# option-pricing/optimal_hedge.py
"""
Mean-Variance hedge helper.

Provides:
 - compute_mv_hedge_ratio_mc: Monte Carlo estimate of hedge ratio h that minimizes Var(L - h * S)
   where L is the option payoff payoff(S_T) discounted, and S is underlying terminal value (or returns).
 - compute_mv_hedge_ratio_historical: uses historical returns/price series to compute covariance-based hedge.

Notes:
 - The hedge ratio returned is the number of underlying units to hold per short option (positive -> long underlying to hedge short option).
"""

import numpy as np
from typing import Optional, Tuple
from .hedging import simulate_black_scholes_paths, black_scholes_price

def compute_mv_hedge_ratio_mc(S0: float, strike: float, r: float, sigma_true: float, T: float,
                              option_type: str = 'call', n_paths: int = 20000, n_steps: int = 50, seed: int = 123):
    """
    Estimate hedge ratio by Monte Carlo under true dynamics:
      - simulate terminal S_T paths
      - compute option payoff per path (discounted if needed)
      - compute covariance between payoff and S_T and return h = Cov(payoff, S_T)/Var(S_T)

    Returns hedge_ratio, and (cov, var) for diagnostics.
    """
    paths = simulate_black_scholes_paths(S0, r, sigma_true, T, n_paths=n_paths, n_steps=n_steps, seed=seed)
    ST = paths[:, -1]
    if option_type == 'call':
        payoff = np.maximum(ST - strike, 0.0)
    else:
        payoff = np.maximum(strike - ST, 0.0)
    # discount payoff to present (optional, hedge ratio usually in asset units which is scale-invariant)
    payoff_disc = payoff * np.exp(-r * T)
    cov = np.cov(payoff_disc, ST, ddof=1)
    cov_payoff_ST = cov[0,1]
    var_ST = cov[1,1]
    if var_ST <= 0:
        return 0.0, (cov_payoff_ST, var_ST)
    h = cov_payoff_ST / var_ST
    return float(h), (float(cov_payoff_ST), float(var_ST))

def compute_mv_hedge_ratio_historical(payoff_series: np.ndarray, underlying_series: np.ndarray):
    """
    Hedge ratio using historical sample covariance:
      h = Cov(payoff, S) / Var(S)
    payoff_series and underlying_series should align and be same length.
    """
    payoff = np.asarray(payoff_series, dtype=float)
    S = np.asarray(underlying_series, dtype=float)
    if payoff.size < 2 or S.size < 2:
        return 0.0
    cov = np.cov(payoff, S, ddof=1)
    cov_payoff_ST = cov[0,1]
    var_ST = cov[1,1]
    if var_ST <= 0:
        return 0.0
    return float(cov_payoff_ST / var_ST)
