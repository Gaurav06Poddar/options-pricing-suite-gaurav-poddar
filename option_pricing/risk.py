# option-pricing/risk.py
"""
Risk metrics module: VaR, CVaR, Greeks aggregation, and scenario testing.

Functions:
- historical_var(pnl_series, alpha)
- gaussian_var(mean, std, alpha)
- cvar_from_samples(pnl_samples, alpha)
- mc_var_from_simulated_pnl(simulator_fn, n_sim, alpha)
- aggregate_portfolio_greeks(portfolio_positions)
- scenario_loss(portfolio, shock_spot_pct, shock_vol_pct, shock_r)
"""

import numpy as np
import pandas as pd
from typing import Callable, Sequence, Dict


# ========================
# VaR & CVaR Calculations
# ========================

def historical_var(pnl_series: Sequence[float], alpha: float = 0.05) -> float:
    """
    Historical VaR at level alpha (e.g., 0.05 -> 5% VaR).
    Returns positive loss figure.
    """
    arr = np.asarray(pnl_series, dtype=float)
    if arr.size == 0:
        return 0.0
    q = np.quantile(arr, alpha)
    return float(-q if q < 0 else 0.0)


def cvar_from_samples(pnl_samples: Sequence[float], alpha: float = 0.05) -> float:
    """
    Conditional VaR (Expected Shortfall) from PnL samples.
    Returns positive loss figure.
    """
    arr = np.asarray(pnl_samples, dtype=float)
    if arr.size == 0:
        return 0.0
    q = np.quantile(arr, alpha)
    tail = arr[arr <= q]
    if tail.size == 0:
        return 0.0
    es = -tail.mean() if tail.mean() < 0 else 0.0
    return float(es)


def gaussian_var(mean: float, std: float, alpha: float = 0.05) -> float:
    """
    Parametric Gaussian VaR.
    Returns positive loss figure.
    """
    from scipy.stats import norm
    z = norm.ppf(alpha)
    var = -(mean + z * std)
    return float(var if var > 0 else 0.0)


def mc_var_from_simulated_pnl(simulator_fn: Callable[[], np.ndarray], n_sim: int = 5000, alpha: float = 0.05) -> dict:
    """
    Monte Carlo VaR & CVaR from a simulator function.
    simulator_fn: returns 1D array or scalar PnL for a single sim.
    Returns dict with {'var', 'cvar', 'samples'}.
    """
    samples = []
    for _ in range(n_sim):
        s = simulator_fn()
        if hasattr(s, "__len__"):
            s = np.asarray(s).mean()
        samples.append(float(s))
    samples = np.asarray(samples, dtype=float)
    return {
        "var": historical_var(samples, alpha),
        "cvar": cvar_from_samples(samples, alpha),
        "samples": samples
    }


# ========================
# Portfolio Risk Helpers
# ========================

def aggregate_portfolio_greeks(portfolio_positions: Sequence[Dict[str, float]]) -> dict:
    """
    Aggregates Delta, Gamma, Vega, Theta, Rho across portfolio.
    portfolio_positions: iterable of dicts with greek keys {'delta','gamma','vega','theta','rho','quantity'}.
    Returns dict with portfolio-level Greeks.
    """
    total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for pos in portfolio_positions:
        qty = pos.get("quantity", 1.0)
        for greek in total.keys():
            total[greek] += qty * pos.get(greek, 0.0)
    return total


def scenario_loss(portfolio, shock_spot_pct=0.0, shock_vol_pct=0.0, shock_r=0.0):
    """
    Computes portfolio PnL under a specified scenario shock.
    Delegates to Portfolio.scenario_shock.
    """
    return portfolio.scenario_shock(
        shock_spot_pct=shock_spot_pct,
        shock_vol_pct=shock_vol_pct,
        shock_r=shock_r
    )


# ========================
# Live-Compatible Risk Wrappers
# ========================

def rolling_var(pnl_series: pd.Series, window: int = 252, alpha: float = 0.05) -> pd.Series:
    """
    Rolling historical VaR for live monitoring.
    Returns a Series indexed like pnl_series.
    """
    return pnl_series.rolling(window).apply(lambda x: historical_var(x, alpha), raw=False)


def rolling_cvar(pnl_series: pd.Series, window: int = 252, alpha: float = 0.05) -> pd.Series:
    """
    Rolling CVaR (Expected Shortfall) for live monitoring.
    Returns a Series indexed like pnl_series.
    """
    return pnl_series.rolling(window).apply(lambda x: cvar_from_samples(x, alpha), raw=False)
