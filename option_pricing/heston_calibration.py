# option-pricing/heston_calibration.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import least_squares

from .BlackScholesModel import BlackScholesModel
from .HestonModel import HestonModel

@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    v0: float

def heston_implied_vol_curve(S0, r, T, strikes, params: HestonParams):
    """
    Given Heston params, produce the Heston European call prices and invert to BS IVs.
    """
    ivs = []
    for K in strikes:
        pricer = HestonModel(S0, K, T, r, params.kappa, params.theta, params.sigma_v, params.rho, params.v0)
        c = pricer.call_price()
        # invert to IV
        bsm = BlackScholesModel(S0, K, int(round(T * 365)), r, 0.2)
        try:
            iv = bsm.implied_volatility(c, 'call')
        except Exception:
            iv = np.nan
        ivs.append(iv)
    return np.array(ivs, dtype=float)

def calibrate_heston_to_iv(
    S0, r, T, strikes, market_ivs,
    init=HestonParams(1.5, 0.04, 0.5, -0.6, 0.04),
    bounds=((1e-4, 1e-4, 1e-4, -0.999, 1e-5),
            (10.0, 2.0, 3.0,  0.999,  2.0))
):
    """
    Least squares fit of Heston parameters to market IVs for a single maturity.
    Returns (params, result) where params is HestonParams and result is scipy OptimizeResult.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_ivs = np.asarray(market_ivs, dtype=float)

    mask = ~np.isnan(market_ivs)
    strikes = strikes[mask]
    market_ivs = market_ivs[mask]
    if len(strikes) < 4:
        raise ValueError("Not enough valid IV points to calibrate.")

    x0 = np.array([init.kappa, init.theta, init.sigma_v, init.rho, init.v0], dtype=float)
    lb, ub = np.array(bounds[0], float), np.array(bounds[1], float)

    def resid(x):
        p = HestonParams(*x.tolist())
        model_ivs = heston_implied_vol_curve(S0, r, T, strikes, p)
        return (model_ivs - market_ivs)

    res = least_squares(resid, x0, bounds=(lb, ub), xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=200)
    fitted = HestonParams(*res.x.tolist())
    return fitted, res

def summarize_fit(strikes, market_ivs, model_ivs):
    df = pd.DataFrame({
        'strike': strikes,
        'market_iv': market_ivs,
        'model_iv': model_ivs,
        'diff': model_ivs - market_ivs
    }).sort_values('strike')
    rmse = np.sqrt(np.nanmean((df['diff'])**2))
    mae = np.nanmean(np.abs(df['diff']))
    return df, rmse, mae
