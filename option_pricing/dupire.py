# option-pricing/dupire.py
"""
Compute Dupire local volatility surface from a grid of implied volatilities.

Inputs:
 - strikes: 1D array of strikes (sorted ascending)
 - maturities: 1D array of days or years (we expect years), must be increasing
 - implied_vols: 2D array shape (len(maturities), len(strikes)) in decimals (annualized)
 - S0: spot for option price conversion (used to compute forward/discounting)
 - r: risk-free rate (annual decimal)

Returns:
 - local_vols: 2D array same shape with local vol at (T,K) (NaN where unstable)
 - price_surface: underlying call price surface used for derivatives (for debugging)
"""

import numpy as np
from .BlackScholesModel import BlackScholesModel

def implied_vols_to_call_prices(strikes, maturities, implied_vols, S0, r):
    """
    Convert implied vol grid (maturities in years) to call prices via Black-Scholes.
    implied_vols shape: (nT, nK)
    Returns price_surface same shape.
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    iv = np.asarray(implied_vols, dtype=float)
    nT, nK = iv.shape
    price = np.full((nT, nK), np.nan, dtype=float)
    for i in range(nT):
        T = float(maturities[i])
        for j in range(nK):
            sigma = float(iv[i, j])
            try:
                bsm = BlackScholesModel(S0, strikes[j], int(max(1, round(T*365))), r, sigma)
                # BlackScholesModel constructor expects days_to_maturity in days; convert T years -> days
                price[i, j] = bsm.calculate_option_price('Call Option')
            except Exception:
                price[i, j] = np.nan
    return price

def dupire_local_vol(strikes, maturities, implied_vols, S0, r=0.0, eps=1e-6):
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)  # in years
    iv = np.asarray(implied_vols, dtype=float)
    nT, nK = iv.shape
    # compute price surface
    price = implied_vols_to_call_prices(strikes, maturities, iv, S0, r)
    # allocate local vol
    local_vol2 = np.full_like(price, np.nan, dtype=float)  # squared local vol
    # compute derivatives by central differences
    # dC/dT
    for i in range(nT):
        for j in range(nK):
            # dC/dT
            if i == 0:
                # forward diff
                if nT > 1:
                    dC_dT = (price[i+1, j] - price[i, j]) / max(maturities[i+1] - maturities[i], eps)
                else:
                    dC_dT = np.nan
            elif i == nT - 1:
                dC_dT = (price[i, j] - price[i-1, j]) / max(maturities[i] - maturities[i-1], eps)
            else:
                dC_dT = (price[i+1, j] - price[i-1, j]) / max(maturities[i+1] - maturities[i-1], eps)

            # dC/dK (first derivative wrt strike)
            if j == 0:
                dC_dK = (price[i, j+1] - price[i, j]) / max(strikes[j+1] - strikes[j], eps)
            elif j == nK - 1:
                dC_dK = (price[i, j] - price[i, j-1]) / max(strikes[j] - strikes[j-1], eps)
            else:
                dC_dK = (price[i, j+1] - price[i, j-1]) / max(strikes[j+1] - strikes[j-1], eps)

            # d2C/dK2 (second derivative)
            if 0 < j < nK - 1:
                dk_forward = strikes[j+1] - strikes[j]
                dk_backward = strikes[j] - strikes[j-1]
                # central second diff with possibly non-uniform grid
                d2C_dK2 = 2.0 * ( (price[i, j+1] - price[i, j]) / (dk_forward * (dk_forward + dk_backward)) - (price[i, j] - price[i, j-1]) / (dk_backward * (dk_forward + dk_backward)) )
            else:
                # forward/backward approx
                d2C_dK2 = np.nan

            # Dupire formula: local variance
            K = strikes[j]
            denom = (K**2) * d2C_dK2 if (d2C_dK2 is not None and not np.isnan(d2C_dK2)) else np.nan
            if denom is None or np.isnan(denom) or abs(denom) < eps:
                local_vol2[i, j] = np.nan
            else:
                local_vol2[i, j] = max(0.0, 2.0 * (dC_dT + r * K * dC_dK) / denom)

    # local vol surface (sqrt), mask invalids
    local_vol = np.sqrt(np.where(local_vol2 >= 0.0, local_vol2, np.nan))
    return local_vol, price
