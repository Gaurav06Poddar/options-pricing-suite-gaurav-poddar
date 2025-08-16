# option-pricing/carr_madan_heston.py
"""
Carr-Madan pricing wrapper for Heston model.

Provides:
- heston_characteristic_function_factory(S0, r, kappa, theta, sigma_v, rho, v0, T)
- carr_madan_price_heston(...) to run FFT with Heston CF and produce call prices vector
"""

import numpy as np
from math import log
from .carr_madan import carr_madan_price

def heston_char_func_factory(S0, r, kappa, theta, sigma_v, rho, v0, T):
    """
    Returns function phi(u) = E[e^{i u ln S_T}] for Heston under risk-neutral measure.
    Implementation follows common Heston char func convention; this returns
    the CF of log S_T (natural log).
    """
    # parameters
    x0 = np.log(S0)
    kappa = float(kappa); theta = float(theta); sigma = float(sigma_v); rho = float(rho); v0 = float(v0)
    r = float(r)

    def phi(u):
        # u can be real or complex
        iu = 1j * u
        a = kappa * theta
        b = kappa
        # following standard Heston char func (Gatheral/Lewis convention)
        d = np.sqrt((rho * sigma * iu - b)**2 + (sigma**2) * (iu + u**2))
        g = (b - rho * sigma * iu - d) / (b - rho * sigma * iu + d)
        # avoid division by zero / numerical issues
        exp_dt = np.exp(-d * T)
        # C and D functions
        C = r * iu * T + (a / (sigma**2)) * ((b - rho * sigma * iu - d) * T - 2.0 * np.log((1 - g * exp_dt) / (1 - g)))
        D = ((b - rho * sigma * iu - d) / (sigma**2)) * ((1 - exp_dt) / (1 - g * exp_dt))
        val = np.exp(C + D * v0 + iu * x0)
        return val
    return phi

def carr_madan_price_heston(S0, r, T, kappa, theta, sigma_v, rho, v0, alpha=1.5, N=2**12, B=800.0):
    phi = heston_char_func_factory(S0, r, kappa, theta, sigma_v, rho, v0, T)
    k_grid, K_grid, prices = carr_madan_price(phi, S0, r, T, alpha=alpha, N=N, B=B)
    return k_grid, K_grid, prices
