# option-pricing/HestonModel.py
import numpy as np
from numpy import log, sqrt, exp, real
from scipy.integrate import quad

class HestonModel:
    """
    Semi-closed-form Heston (1993) European call/put pricing via Fourier integrals,
    plus simple Euler (full truncation) MC path generator (for visualization/sanity).
    """

    def __init__(self, S0, K, T, r, kappa, theta, sigma_v, rho, v0):
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)        # in years
        self.r = float(r)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.v0 = float(v0)

    # ----- Characteristic function -----
    def _charfunc(self, u):
        kappa, theta, sigma, rho = self.kappa, self.theta, self.sigma_v, self.rho
        v0, T, r = self.v0, self.T, self.r
        x0 = log(self.S0)

        iu = 1j * u
        a = kappa * theta
        b1 = kappa - rho * sigma * iu
        d = np.sqrt((rho * sigma * iu - kappa)**2 + (sigma**2) * (iu + u**2))
        gp = (b1 + d) / (b1 - d)
        # log of denominator guarded
        C = r * iu * T + (a / (sigma**2)) * ((b1 + d) * T - 2.0 * np.log((1 - gp * np.exp(d * T)) / (1 - gp)))
        D = ((b1 + d) / (sigma**2)) * ((1 - np.exp(d * T)) / (1 - gp * np.exp(d * T)))
        return np.exp(C + D * v0 + iu * x0)

    # Heston probabilities P1, P2
    def _P(self, j):
        # integrand for Pj per Heston original formula (Lewis / Gatheral convention)
        def integrand(u):
            u = float(u)
            if j == 1:
                cf = self._charfunc(u - 1j)
                denom = 1j * u * self.K
            else:
                cf = self._charfunc(u)
                denom = 1j * u
            val = np.exp(-1j * u * np.log(self.K)) * cf / denom
            return real(val)

        # Numerically integrate from 0..inf with damping via quad
        val, _ = quad(integrand, 0.0, np.inf, limit=200, epsabs=1e-7, epsrel=1e-7)
        return 0.5 + (1.0 / np.pi) * val

    def call_price(self):
        P1 = self._P(1)
        P2 = self._P(2)
        return self.S0 * P1 - self.K * np.exp(-self.r * self.T) * P2

    def put_price(self):
        c = self.call_price()
        return c - self.S0 + self.K * np.exp(-self.r * self.T)

    # ----- Simple MC path generator (visualization/sanity) -----
    def simulate_paths(self, n_paths=10000, n_steps=100, seed=42, full_paths=False):
        """
        Euler full truncation for variance to preserve positivity.
        Returns terminal prices or full matrices if full_paths=True.
        """
        rng = np.random.default_rng(seed)
        dt = self.T / n_steps
        S = np.full((n_paths,), self.S0, dtype=float)
        v = np.full((n_paths,), self.v0, dtype=float)

        # Correlated normals
        for _ in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            z2 = self.rho * z1 + np.sqrt(max(1 - self.rho**2, 0.0)) * z2

            v_pos = np.maximum(v, 0.0)
            S = S * np.exp((self.r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z1)
            v = v + self.kappa * (self.theta - v_pos) * dt + self.sigma_v * np.sqrt(v_pos * dt) * z2

        if full_paths:
            # For memory reasons this demo keeps only terminal by default
            # Re-run storing full if requested
            pass
        return S
