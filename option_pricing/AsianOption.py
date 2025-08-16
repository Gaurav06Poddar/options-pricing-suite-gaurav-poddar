"""
Arithmetic-average Asian option pricing (Monte Carlo).
Simple path simulation; supports control variate using geometric Asian (closed-form)
for variance reduction when desired.
"""

import numpy as np
from math import exp, log, sqrt
from scipy.stats import norm


class AsianMonteCarlo:
    """
    Arithmetic Asian option Monte Carlo pricing.

    Parameters:
    - S0, K, r, sigma, days, n_paths, n_steps, payoff ('call'|'put')
    - variance_reduction: 'none' or 'control' (control using geometric Asian closed-form)
    """

    def __init__(self, S0, K, r, sigma, days, n_paths=20000, n_steps=None, payoff='call', variance_reduction='none', seed=1234):
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.days = int(days)
        self.T = float(days) / 365.0
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps) if n_steps is not None else max(1, self.days)
        self.dt = self.T / float(self.n_steps)
        self.payoff = payoff.lower()
        self.variance_reduction = variance_reduction.lower()
        self.seed = seed
        self.paths = None

    def _simulate_paths(self):
        rng = np.random.default_rng(self.seed)
        steps = self.n_steps
        dt = self.dt
        paths = np.zeros((steps + 1, self.n_paths))
        paths[0, :] = self.S0
        for t in range(1, steps + 1):
            Z = rng.standard_normal(self.n_paths)
            paths[t, :] = paths[t - 1, :] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z)
        self.paths = paths

    def _payoff_from_paths(self):
        # arithmetic average excluding time 0 or including? We'll include all times (0..T)
        avg = np.mean(self.paths, axis=0)
        if self.payoff == 'call':
            payoff = np.maximum(avg - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - avg, 0.0)
        return payoff

    def _geometric_asian_price(self):
        """
        Closed-form price for geometric-average Asian option (discrete sampling equally spaced).
        Uses known closed-form formula for geometric Asian option.
        """
        n = self.n_steps + 1  # sampling points including t=0
        dt = self.dt
        T = self.T
        sigma = self.sigma
        r = self.r
        S0 = self.S0
        K = self.K

        # effective parameters for geometric average
        sigma_g = sigma * sqrt((2 * n + 1) / (6.0 * (n + 1)))
        mu_g = 0.5 * (r - 0.5 * sigma ** 2) * (n) / (n + 1) + 0.5 * sigma ** 2 * (n) / (n + 1)
        # actually simpler approach: use continuous approximation (ok for demo)
        mu_g = r - 0.5 * sigma ** 2 * (n - 1) / (n + 1)
        sigma_g = sigma * sqrt((n + 1) * (2 * n + 1) / (6.0 * n ** 2))

        # Use Black-Scholes formula on geometric average proxy
        S_adj = S0 * np.exp(mu_g * T)
        d1 = (np.log(S_adj / K) + 0.5 * sigma_g ** 2 * T) / (sigma_g * np.sqrt(T) + 1e-12)
        d2 = d1 - sigma_g * np.sqrt(T)
        if self.payoff == 'call':
            price = np.exp(-r * T) * (S_adj * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = np.exp(-r * T) * (K * norm.cdf(-d2) - S_adj * norm.cdf(-d1))
        return float(price)

    def price(self):
        self._simulate_paths()
        payoff = self._payoff_from_paths()
        disc = np.exp(-self.r * self.T) * np.mean(payoff)
        if self.variance_reduction == 'control':
            # control: geometric Asian closed-form
            geo_price = self._geometric_asian_price()
            geo_payoff = np.exp(-self.r * self.T) * np.maximum(np.exp(np.mean(np.log(self.paths + 1e-12), axis=0)) - self.K, 0.0)
            cov = np.cov(payoff, geo_payoff, bias=True)[0, 1]
            var_geo = np.var(geo_payoff)
            if var_geo > 0:
                c = cov / var_geo
                adj = payoff - c * (geo_payoff - geo_price)
                disc = np.exp(-self.r * self.T) * np.mean(adj)
        return float(disc)
