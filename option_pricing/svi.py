#option-pricing-models/option-pricing/svi.py
"""
SVI (Stochastic Volatility Inspired) parametric fit for implied variance w(k) = total implied variance.
Reference: Gatheral (2004), "The Volatility Surface: A Practitioner's Guide".

Parameterization (raw SVI):
w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))
where k = log(K/F), total variance w = iv^2 * T

This module:
- fits SVI parameters (a,b,rho,m,sigma) per expiry using least squares
- returns smoothed IVs (interpolated) on a strike grid
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def fit_svi_for_expiry(strikes, ivs, F, T, initial=None, bounds=None):
    """
    Fit SVI for a single expiry.

    strikes: array of strike prices K
    ivs: implied vols (decimal)
    F: forward price (approx spot*exp(rT) or market forward)
    T: time to expiry in years

    Returns: params dict {a,b,rho,m,sigma} and success flag
    """
    K = np.asarray(strikes, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    mask = np.isfinite(ivs) & (ivs > 0)
    if mask.sum() < 5:
        raise ValueError("Not enough IV points to fit SVI (need >=5).")
    K = K[mask]
    ivs = ivs[mask]
    k = np.log(K / float(F))
    total_var = (ivs**2) * T

    # initial guess
    if initial is None:
        a0 = np.min(total_var) * 0.5
        b0 = max(0.1, np.std(total_var) / 2.0)
        rho0 = 0.0
        m0 = 0.0
        sigma0 = 0.1
        x0 = np.array([a0, b0, rho0, m0, sigma0])
    else:
        x0 = np.array([initial.get(k, 0.0) for k in ['a','b','rho','m','sigma']], dtype=float)

    if bounds is None:
        lb = [-1.0, 1e-8, -0.999, -5.0, 1e-8]
        ub = [5.0, 5.0, 0.999, 5.0, 5.0]
    else:
        lb, ub = bounds

    def resid(x):
        a,b,rho,m,sig = x
        model = svi_total_variance(k, a,b,rho,m,sig)
        return (model - total_var)

    res = least_squares(resid, x0, bounds=(lb,ub), xtol=1e-9, ftol=1e-9, max_nfev=5000)
    params = {'a': float(res.x[0]), 'b': float(res.x[1]), 'rho': float(res.x[2]), 'm': float(res.x[3]), 'sigma': float(res.x[4])}
    return params, res

def svi_smoothed_iv_grid(strikes_grid, F, T, params):
    """
    Given a grid of strikes K, produce implied vol estimates from SVI params.
    Returns iv_grid (same shape as strikes_grid).
    """
    K = np.asarray(strikes_grid, dtype=float)
    k = np.log(K / float(F))
    w = svi_total_variance(k, params['a'], params['b'], params['rho'], params['m'], params['sigma'])
    # ensure non-negative variance
    w = np.clip(w, 1e-12, None)
    iv = np.sqrt(w / T)
    return iv
