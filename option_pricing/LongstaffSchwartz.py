"""
Longstaff-Schwartz implementation for American option pricing (Monte Carlo / least-squares).
Uses polynomial basis (1, S, S^2, ...) for the continuation value regression.
"""

import numpy as np


class LongstaffSchwartz:
    """
    Longstaff-Schwartz pricing engine.

    Parameters:
    - S0: spot
    - K: strike
    - r: risk-free rate (annual)
    - sigma: vol (annual)
    - days: days to maturity
    - n_paths: number of MC paths
    - n_steps: time steps in path (use days or fewer)
    - payoff: 'call' or 'put'
    - poly_degree: degree of polynomial regression basis
    """

    def __init__(self, S0, K, r, sigma, days, n_paths=20000, n_steps=None, payoff='put', poly_degree=2, seed=12345):
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.days = int(days)
        self.T = float(days) / 365.0
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps) if n_steps is not None else max(1, self.days)
        self.dt = self.T / float(self.n_steps) if self.n_steps > 0 else 0.0
        self.payoff = payoff.lower()
        self.poly_degree = int(poly_degree)
        self.seed = seed

        # Will be filled by simulate_paths
        self.paths = None  # shape (n_steps+1, n_paths)
        self.cashflows = None

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

    def _intrinsic(self, S):
        if self.payoff == 'call':
            return np.maximum(S - self.K, 0.0)
        else:
            return np.maximum(self.K - S, 0.0)

    def price(self):
        """
        Run LSM and return estimated American option price (and optionally std-error).
        """
        self._simulate_paths()
        steps = self.n_steps
        paths = self.paths
        dt = self.dt
        df = np.exp(-self.r * dt)

        # cashflow matrix: each path has eventual cashflow; fill with payoff at maturity
        cashflow = self._intrinsic(paths[-1, :]).copy()
        # store time-of-exercise indicator if needed
        for t in range(steps - 1, 0, -1):  # backwards in time, exclude t=0 (start)
            S_t = paths[t, :]
            immediate_ex = self._intrinsic(S_t)
            # find in-the-money paths
            itm = immediate_ex > 0
            if not np.any(itm):
                # no paths ITM at this time, continue
                cashflow = cashflow * df
                continue

            # discounted cashflow to time t
            discounted = cashflow * df

            # regression basis on S_t[itm]
            X = np.vander(S_t[itm], N=self.poly_degree + 1, increasing=True)  # columns 1, S, S^2, ...
            y = discounted[itm]

            # linear regression via least squares
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            # estimate continuation value for ITM paths
            cont = X.dot(coeffs)

            # decide exercise where immediate_ex > continuation
            exercise_now = immediate_ex[itm] > cont
            # update cashflow: where exercise_now True, set cashflow to immediate_ex; else discounted continuation remains
            exercise_indices = np.where(itm)[0][exercise_now]
            cashflow[exercise_indices] = immediate_ex[exercise_indices]
            # discount remaining cashflow to previous time step
            cashflow = cashflow * df

        # At time 0, discount to present value
        price = np.mean(cashflow)  # cashflow already discounted through loop? last loop discounted to t=0
        return float(price)
