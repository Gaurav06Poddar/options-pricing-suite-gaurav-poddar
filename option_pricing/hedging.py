# option-pricing/hedging.py
"""
Dynamic hedging backtest (multi-rebalance) and helpers.

Provides:
- simulate_black_scholes_paths(S0, r, sigma, T, n_paths, n_steps, seed)
- black_scholes_delta(S, K, r, sigma, T, option_type='call')
- black_scholes_price(S, K, r, sigma, T, option_type='call')
- backtest_dynamic_hedge(paths, strike, r, sigma_model, T, option_type='call',
                         rebalance_every=1, tc_per_share=0.0, impact_per_share=0.0,
                         impact_coeff_quadratic=0.0, liquidity_scale=1e6,
                         use_model_price_for_init=True)

Behavior summary:
- Short one option written at t0 (we receive premium = model price if use_model_price_for_init True).
- Maintain a dynamic hedge: long delta_t shares (computed with BSM delta using sigma_model).
- Cash account accrues at continuous compounding at rate r.
- Transaction costs applied on share trades: fixed per-share cost, linear slippage, and optional quadratic impact.
- Returns per-path P&L (discounted) and summary/details dict.
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy.stats import norm

# ------------------------
# Path simulation
# ------------------------
def simulate_black_scholes_paths(S0: float, r: float, sigma: float, T: float,
                                 n_paths: int = 10000, n_steps: int = 252, seed: int = 1234) -> np.ndarray:
    """
    Simulate geometric Brownian motion price paths (S_t) using vectorized operations.

    Returns array of shape (n_paths, n_steps+1) including t=0.
    """
    rng = np.random.default_rng(seed)
    dt = float(T) / float(n_steps)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = float(S0)
    for t in range(1, n_steps + 1):
        Z = rng.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

# ------------------------
# Black-Scholes helpers
# ------------------------
def _d1(S, K, r, sigma, T):
    S = np.asarray(S, dtype=float)
    eps = 1e-12
    T = np.maximum(T, eps)
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + eps)

def black_scholes_delta(S, K, r, sigma, T, option_type='call'):
    """
    Vectorized Black-Scholes delta for European call/put.
    S can be scalar or ndarray, T can be scalar or ndarray (years).
    """
    d1 = _d1(S, K, r, sigma, T)
    Nd1 = norm.cdf(d1)
    if option_type == 'call':
        return Nd1
    else:
        return Nd1 - 1.0

def black_scholes_price(S, K, r, sigma, T, option_type='call'):
    """
    Vectorized Black-Scholes price. Handles scalar / array S and T (years).
    """
    eps = 1e-12
    T = np.maximum(T, eps)
    d1 = _d1(S, K, r, sigma, T)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# ------------------------
# Dynamic hedging backtest
# ------------------------
def backtest_dynamic_hedge(paths: np.ndarray,
                           strike: float,
                           r: float,
                           sigma_model: float,
                           T: float,
                           option_type: str = 'call',
                           rebalance_every: int = 1,
                           tc_per_share: float = 0.0,
                           impact_per_share: float = 0.0,
                           impact_coeff_quadratic: float = 0.0,
                           liquidity_scale: float = 1e6,
                           use_model_price_for_init: bool = True
                           ) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Dynamic hedging backtest with enhanced execution-cost model.

    Args:
      paths: ndarray (n_paths, n_steps+1) of simulated underlying prices including t0.
      strike: option strike price.
      r: risk-free rate (annual decimal).
      sigma_model: volatility used for model (to compute deltas/prices).
      T: time to maturity in years.
      option_type: 'call' or 'put'.
      rebalance_every: rebalance frequency in time-steps (1 => rebalance every step).
      tc_per_share: fixed per-share transaction cost (e.g., commission/spread).
      impact_per_share: linear slippage per share ($).
      impact_coeff_quadratic: coefficient for quadratic impact tied to trade notional.
      liquidity_scale: notional normalization for quadratic term (e.g., 1e6).
      use_model_price_for_init: if True, we credit initial premium = model price (deterministic at S0 mean).

    Returns:
      df: DataFrame with per-path final metrics and discounted pnl ('pnl_disc').
      summary: dict with mean/std/median/quantiles of pnl_disc.
      details: dict with run parameters and initial premium info.
    """
    # Validate inputs
    if paths.ndim != 2:
        raise ValueError("paths must be a 2D ndarray of shape (n_paths, n_steps+1)")

    n_paths, n_steps_p1 = paths.shape
    n_steps = n_steps_p1 - 1
    if n_steps <= 0:
        raise ValueError("paths must contain at least one time step (shape[1] >= 2)")

    dt = float(T) / float(n_steps)
    times = np.linspace(0.0, float(T), n_steps + 1)

    # initial premium (deterministic) using mean S0 for bookkeeping
    S0_mean = float(np.mean(paths[:, 0]))
    if use_model_price_for_init:
        initial_premium = float(black_scholes_price(S0_mean, strike, r, sigma_model, T, option_type))
    else:
        initial_premium = 0.0

    # prepare arrays
    cash = np.zeros((n_paths, n_steps + 1), dtype=float)
    delta = np.zeros((n_paths, n_steps + 1), dtype=float)

    # initial delta and cash: compute per-path delta at t0 using each path S0 (vector)
    S_t0 = paths[:, 0]
    delta0 = black_scholes_delta(S_t0, strike, r, sigma_model, T, option_type)

    # initial premium treatment: we use deterministic initial_premium for bookkeeping to avoid path-dependent premium inconsistencies
    if use_model_price_for_init:
        premium = initial_premium  # scalar
        # cash0 = premium - cost to buy hedge shares - transaction costs for acquiring shares
        tc_initial = tc_per_share * np.abs(delta0) + impact_per_share * np.abs(delta0) + \
                     impact_coeff_quadratic * ((np.abs(delta0) * S_t0) / float(liquidity_scale))**2
        cash0 = premium - delta0 * S_t0 - tc_initial
    else:
        tc_initial = tc_per_share * np.abs(delta0) + impact_per_share * np.abs(delta0) + \
                     impact_coeff_quadratic * ((np.abs(delta0) * S_t0) / float(liquidity_scale))**2
        cash0 = - delta0 * S_t0 - tc_initial

    cash[:, 0] = cash0
    delta[:, 0] = delta0

    # time stepping with rebalancing
    for t in range(1, n_steps + 1):
        S_t = paths[:, t]
        time_to_maturity = float(max(1e-12, T - times[t]))

        # accrue interest on cash (continuous compounding approx)
        cash[:, t] = cash[:, t-1] * np.exp(r * dt)
        delta[:, t] = delta[:, t-1]

        # decide whether to rebalance now
        if (t % int(max(1, rebalance_every))) == 0 or t == n_steps:
            new_delta = black_scholes_delta(S_t, strike, r, sigma_model, time_to_maturity, option_type)

            trade = new_delta - delta[:, t-1]  # positive => buy shares
            trade_abs = np.abs(trade)

            # transaction cost components per path:
            tc_fixed = tc_per_share * trade_abs
            tc_linear = impact_per_share * trade_abs
            # quadratic impact proportional to (trade_value / liquidity_scale)^2
            trade_value = trade_abs * S_t
            tc_quad = impact_coeff_quadratic * (trade_value / float(liquidity_scale))**2

            total_tc = tc_fixed + tc_linear + tc_quad

            # cash update: buy (or sell) the shares at current price and pay transaction costs
            cash[:, t] = cash[:, t] - trade * S_t - total_tc
            delta[:, t] = new_delta

    # at maturity compute payoff and final P&L
    ST = paths[:, -1]
    if option_type == 'call':
        payoff = np.maximum(ST - strike, 0.0)
    else:
        payoff = np.maximum(strike - ST, 0.0)

    final_cash = cash[:, -1]
    final_delta = delta[:, -1]
    final_stock_value = final_delta * ST

    # final portfolio P&L (we are short one option)
    pnl = final_cash + final_stock_value - payoff

    # discount P&L to present value
    pnl_disc = pnl * np.exp(-r * T)

    # build DataFrame
    df = pd.DataFrame({
        'ST': ST,
        'payoff': payoff,
        'final_cash': final_cash,
        'final_delta': final_delta,
        'final_stock_value': final_stock_value,
        'pnl': pnl,
        'pnl_disc': pnl_disc
    })

    # summary statistics
    summary = {
        'mean_pnl_disc': float(np.mean(pnl_disc)),
        'std_pnl_disc': float(np.std(pnl_disc, ddof=1)) if pnl_disc.size > 1 else 0.0,
        'median_pnl_disc': float(np.median(pnl_disc)),
        'pnl_disc_5pct': float(np.quantile(pnl_disc, 0.05)) if pnl_disc.size > 0 else 0.0,
        'pnl_disc_95pct': float(np.quantile(pnl_disc, 0.95)) if pnl_disc.size > 0 else 0.0,
        'n_paths': int(n_paths),
        'n_steps': int(n_steps),
    }

    details = {
        'initial_premium': float(initial_premium),
        'rebalance_every': int(rebalance_every),
        'tc_per_share': float(tc_per_share),
        'impact_per_share': float(impact_per_share),
        'impact_coeff_quadratic': float(impact_coeff_quadratic),
        'liquidity_scale': float(liquidity_scale),
        'sigma_model': float(sigma_model),
        'T': float(T)
    }

    return df, summary, details
