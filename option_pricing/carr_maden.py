# option-pricing/carr_madan.py
"""
Carr-Madan FFT option pricer.

Reference:
Carr, Madan (1999), "Option Valuation Using the Fast Fourier Transform"

This implements the pricing pipeline:
 - given char. function phi(u) of log-price under risk-neutral measure,
 - compute Fourier transform of modified call price and invert via FFT to obtain call prices.
 
Usage:
 - provide characteristic function cf(u) (complex -> complex)
 - choose parameters: alpha (damping > 0 for calls), eta (grid spacing in freq domain),
   N (number of FFT points), B (upper bound in log-strike domain)
 - get arrays of strikes (K) and call prices C(K).
 
This module provides a Black-Scholes phi for quick testing.
"""

import numpy as np
from typing import Callable, Tuple

def cf_bs(u: np.ndarray, S0: float, r: float, sigma: float, T: float) -> np.ndarray:
    """
    Characteristic function of log S_T for Black-Scholes:
    If X = log S_T, then phi(u) = exp(i u mu - 0.5 sigma^2 u^2)
    with mu = log S0 + (r - 0.5 sigma^2) T
    """
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T
    iu = 1j * u
    return np.exp(iu * mu - 0.5 * (sigma**2) * (u**2) * T)

def carr_madan_fft(cf: Callable[[np.ndarray], np.ndarray],
                   S0: float,
                   r: float,
                   T: float,
                   alpha: float = 1.5,
                   eta: float = 0.25,
                   N: int = 2**12,
                   B: float = 150.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carr-Madan FFT pricing.

    Args:
      cf: characteristic function of log-price, callable on numpy array u
      S0, r, T: model params (S0 used to define strikes region)
      alpha: damping factor > 0 for calls (common 1-2)
      eta: spacing in frequency domain
      N: number of FFT points (power of two recommended)
      B: upper bound for log-strike grid (range ~ [-B, B])

    Returns:
      strikes (numpy array), call_prices (numpy array) aligned such that strikes are increasing.
    """
    # frequency grid
    j = np.arange(N)
    vj = eta * j  # frequencies
    # Simpson weight: first term 1, then alternating 4/2... but Carr-Madan uses weights for integration
    # We'll use trapezoid with simple weights for stability
    # Compute psi(v) = e^{-rT} * cf(v - (alpha+1)i) / (alpha^2 + alpha - v^2 + i(2alpha+1) v)
    i = 1j
    # shifted argument
    arg = vj - i * (alpha + 1.0)
    cf_vals = cf(arg)
    numerator = np.exp(-r * T) * cf_vals
    denom = alpha**2 + alpha - vj**2 + i * (2.0 * alpha + 1.0) * vj
    psi = numerator / denom
    # exponential factor for FFT inversion
    # apply Simpson/trapezoidal weights: here simple trapezoid, weight[0]=0.5, others=1 except last=0.5
    weights = np.ones(N)
    weights[0] = 0.5
    weights[-1] = 0.5
    # function to FFT: e^{i v b} * psi(v) * eta * weight
    b = B / 2.0  # choose b so that log-strike grid ~ [-b, b]
    # Evaluate integrand
    integrand = np.exp(1j * vj * b) * psi * eta * weights
    # FFT
    fft_vals = np.fft.fft(integrand)
    # recovered values correspond to log-strike grid
    km = -b + (2.0 * b / N) * np.arange(N)  # log-strike grid
    strikes = np.exp(km) * S0  # convert log-strike to strike, centered at S0
    # call prices from transform (real part)
    call_vals = np.real(np.exp(-alpha * km) / np.pi * fft_vals)
    # Ensure ordering by strikes ascending
    order = np.argsort(strikes)
    strikes_sorted = strikes[order]
    calls_sorted = call_vals[order]
    return strikes_sorted, calls_sorted
