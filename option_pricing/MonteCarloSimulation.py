# option-pricing/MonteCarloSimulation.py
import numpy as np
import matplotlib.pyplot as plt
from .base import OptionPricingModel

class MonteCarloPricing(OptionPricingModel):
    """
    Monte Carlo pricing with variance-reduction options and low-variance Greeks:
     - antithetic sampling
     - optional control variate (using S_T with known mean)
     - pathwise estimators for Delta/Vega (when valid)
     - likelihood-ratio estimators as fallback
     - CRN central-diff Gamma
     - returns standard errors for estimators
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations=10000, variance_reduction='none'):
        self.S_0 = float(underlying_spot_price)
        self.K = float(strike_price)
        self.days = int(days_to_maturity)
        self.T = float(days_to_maturity) / 365.0
        self.r = float(risk_free_rate)
        self.sigma = float(sigma)

        self.N = int(number_of_simulations)
        self.variance_reduction = variance_reduction.lower() if variance_reduction is not None else 'none'
        self.simulation_terminal = None
        self._seed = 20

    # -----------------------
    # Sampling helpers
    # -----------------------
    def _sample_Z(self, rng):
        if self.variance_reduction in ('antithetic', 'both'):
            half = self.N // 2
            Z_half = rng.standard_normal(half)
            Z = np.concatenate([Z_half, -Z_half])
            if self.N % 2 == 1:
                Z = np.concatenate([Z, rng.standard_normal(1)])
            return Z
        return rng.standard_normal(self.N)

    def _simulate_terminal_prices(self, seed=None):
        if seed is None:
            seed = self._seed
        rng = np.random.default_rng(seed)
        Z = self._sample_Z(rng)
        drift = (self.r - 0.5 * self.sigma ** 2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z
        S_T = self.S_0 * np.exp(drift + diffusion)
        self.simulation_terminal = np.asarray(S_T)

    def simulate_prices(self, seed=None):
        self._simulate_terminal_prices(seed=seed if seed is not None else self._seed)

    # -----------------------
    # Payoffs & pricing
    # -----------------------
    def _payoffs(self, option_type='call'):
        S_T = self.simulation_terminal
        if S_T is None:
            return None
        if option_type == 'call':
            payoff = np.maximum(S_T - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - S_T, 0.0)
        return payoff

    def _control_adjust(self, samples):
        """
        Simple control variate using S_T with known mean E[S_T]=S0*exp(rT).
        samples: raw discounted payoff contributions (not yet averaged)
        """
        if self.variance_reduction in ('control', 'both'):
            S_T = self.simulation_terminal
            disc = np.exp(-self.r * self.T)
            control = disc * (S_T - self.S_0 * np.exp(self.r * self.T))
            cov = np.cov(samples, control, ddof=1)[0, 1]
            var_c = np.var(control, ddof=1)
            beta = 0.0 if var_c <= 0 else cov / var_c
            samples = samples - beta * control
        return samples

    def _price_and_std(self, option_type='call'):
        payoff = self._payoffs(option_type)
        if payoff is None:
            return -1.0, None
        disc = np.exp(-self.r * self.T)
        raw = disc * payoff
        raw = self._control_adjust(raw)
        mean = np.mean(raw)
        std = np.std(raw, ddof=1)
        se = std / np.sqrt(len(raw))
        return float(mean), float(se)

    def _price_from_terminal(self, option_type='call'):
        price, _ = self._price_and_std(option_type)
        return price

    def _calculate_call_option_price(self):
        return self._price_from_terminal('call')

    def _calculate_put_option_price(self):
        return self._price_from_terminal('put')

    # -----------------------
    # Pathwise estimators (low variance)
    # -----------------------
    def _reconstruct_Z(self, S_T):
        eps = 1e-12
        denom = self.sigma * np.sqrt(max(self.T, eps))
        return (np.log(np.maximum(S_T, eps) / self.S_0) - (self.r - 0.5 * self.sigma**2) * self.T) / denom

    def pathwise_delta(self, option_type='call'):
        """
        Legacy API: return scalar pathwise delta estimate (no stderr).
        For stderr use pathwise_delta_se()
        """
        est, se = self.pathwise_delta_se(option_type)
        return float(est)

    def pathwise_delta_se(self, option_type='call'):
        """
        Return (estimate, stderr) for pathwise delta
        """
        if self.simulation_terminal is None:
            self._simulate_terminal_prices(seed=self._seed)
        S_T = self.simulation_terminal
        disc = np.exp(-self.r * self.T)
        if option_type == 'call':
            indicator = (S_T > self.K).astype(float)
            samples = disc * indicator * (S_T / self.S_0)
        else:
            indicator = (S_T < self.K).astype(float)
            samples = disc * (-indicator * (S_T / self.S_0))
        est_samples = self._control_adjust(samples)
        est = np.mean(est_samples)
        se = np.std(est_samples, ddof=1) / np.sqrt(len(est_samples))
        return float(est), float(se)

    def pathwise_vega(self, option_type='call'):
        """
        Legacy API: scalar vega estimate
        """
        est, se = self.pathwise_vega_se(option_type)
        return float(est)

    def pathwise_vega_se(self, option_type='call'):
        """
        Return (estimate, stderr) for pathwise vega
        """
        if self.simulation_terminal is None:
            self._simulate_terminal_prices(seed=self._seed)
        S_T = self.simulation_terminal
        Z = self._reconstruct_Z(S_T)
        disc = np.exp(-self.r * self.T)
        dS_dsigma = S_T * (-self.sigma * self.T + np.sqrt(max(self.T, 1e-12)) * Z)
        if option_type == 'call':
            indicator = (S_T > self.K).astype(float)
            samples = disc * indicator * dS_dsigma
        else:
            indicator = (S_T < self.K).astype(float)
            samples = disc * (-indicator * dS_dsigma)
        est_samples = self._control_adjust(samples)
        est = np.mean(est_samples)
        se = np.std(est_samples, ddof=1) / np.sqrt(len(est_samples))
        return float(est), float(se)


    # -----------------------
    # Likelihood ratio estimators (fallback)
    # -----------------------
    def lr_delta(self, option_type='call'):
        if self.simulation_terminal is None:
            self._simulate_terminal_prices(seed=self._seed)
        S_T = self.simulation_terminal
        disc = np.exp(-self.r * self.T)
        eps = 1e-12
        score = (np.log(np.maximum(S_T, eps) / self.S_0) - (self.r - 0.5 * self.sigma**2) * self.T) / (self.S_0 * self.sigma**2 * max(self.T, eps)) - 1.0 / self.S_0
        payoff = np.maximum(S_T - self.K, 0.0) if option_type == 'call' else np.maximum(self.K - S_T, 0.0)
        samples = disc * payoff * score
        samples = self._control_adjust(samples)
        est = np.mean(samples)
        se = np.std(samples, ddof=1) / np.sqrt(len(samples))
        return float(est), float(se)

    def lr_vega(self, option_type='call'):
        if self.simulation_terminal is None:
            self._simulate_terminal_prices(seed=self._seed)
        S_T = self.simulation_terminal
        disc = np.exp(-self.r * self.T)
        h = 1e-5
        eps = 1e-12
        def logpdf(sigma_val):
            mu = np.log(self.S_0) + (self.r - 0.5 * sigma_val**2) * self.T
            var = max((sigma_val**2) * self.T, 1e-12)
            lt = np.log(np.maximum(S_T, eps))
            return -0.5 * ((lt - mu)**2) / var - 0.5 * np.log(2.0 * np.pi * var) - lt
        score = (logpdf(self.sigma + h) - logpdf(max(self.sigma - h, 1e-8))) / (2 * h)
        payoff = np.maximum(S_T - self.K, 0.0) if option_type == 'call' else np.maximum(self.K - S_T, 0.0)
        samples = disc * payoff * score
        samples = self._control_adjust(samples)
        est = np.mean(samples)
        se = np.std(samples, ddof=1) / np.sqrt(len(samples))
        return float(est), float(se)

    # -----------------------
    # Central-diff Gamma with CRN + antithetics
    # -----------------------
    def gamma_central_diff(self, option_type='call', eps=1e-2):
        rng = np.random.default_rng(self._seed)
        Z = self._sample_Z(rng)

        def price_with_S0(S0):
            ST = S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
            payoff = np.maximum(ST - self.K, 0.0) if option_type == 'call' else np.maximum(self.K - ST, 0.0)
            disc = np.exp(-self.r * self.T)
            raw = disc * payoff
            # control adjust if requested
            if self.variance_reduction in ('control', 'both'):
                control = disc * (ST - S0 * np.exp(self.r * self.T))
                cov = np.cov(raw, control, ddof=1)[0, 1]
                var_c = np.var(control, ddof=1)
                beta = 0.0 if var_c <= 0 else cov / var_c
                raw = raw - beta * control
            return np.mean(raw)

        p0 = price_with_S0(self.S_0)
        p_plus = price_with_S0(self.S_0 + eps)
        p_minus = price_with_S0(max(self.S_0 - eps, 1e-12))
        gamma = (p_plus - 2 * p0 + p_minus) / (eps**2)

        # simple SE proxy from the three evaluations
        se = np.std([p_plus, p0, p_minus], ddof=1) / (abs(eps)**2)
        return float(gamma), float(se)

    # -----------------------
    # Combined API
    # -----------------------
    def greeks_with_errors(self, option_type='call', use_pathwise=True):
        if self.simulation_terminal is None:
            self._simulate_terminal_prices(seed=self._seed)

        out = {}
        price, price_se = self._price_and_std(option_type)
        out['price'] = (price, price_se)

        if use_pathwise:
            try:
                out['delta'] = self.pathwise_delta_se(option_type)
            except Exception:
                out['delta'] = self.lr_delta(option_type)
            try:
                out['vega'] = self.pathwise_vega_se(option_type)
            except Exception:
                out['vega'] = self.lr_vega(option_type)
        else:
            out['delta'] = self.lr_delta(option_type)
            out['vega'] = self.lr_vega(option_type)

        out['gamma'] = self.gamma_central_diff(option_type)
        out['theta'] = (self.theta(option_type), 0.0)
        out['rho']   = (self.rho(option_type), 0.0)
        return out

    # -----------------------
    # Scalar greeks (legacy API)
    # -----------------------
    def delta(self, option_type='call', eps=1e-2):
        est, _ = self.pathwise_delta(option_type)
        return est

    def gamma(self, option_type='call', eps=1e-2):
        g, _ = self.gamma_central_diff(option_type, eps=eps)
        return g

    def vega(self, option_type='call', eps=1e-4):
        est, _ = self.pathwise_vega(option_type)
        return est

    def rho(self, option_type='call', eps=1e-4):
        base = self._price_from_terminal(option_type)
        # CRN bump on r
        backup_r = self.r
        self.r = backup_r + eps
        self._simulate_terminal_prices(seed=self._seed)
        p_plus = self._price_from_terminal(option_type)
        self.r = backup_r - eps
        self._simulate_terminal_prices(seed=self._seed)
        p_minus = self._price_from_terminal(option_type)
        self.r = backup_r
        self._simulate_terminal_prices(seed=self._seed)
        return 0.5 * (p_plus - p_minus) / eps

    def theta(self, option_type='call', days_eps=1):
        if self.days - days_eps <= 0:
            return 0.0
        base = self._price_from_terminal(option_type)
        backup_days = self.days
        self.days = max(self.days - days_eps, 1)
        self.T = self.days / 365.0
        self._simulate_terminal_prices(seed=self._seed)
        p_next = self._price_from_terminal(option_type)
        # restore
        self.days = backup_days
        self.T = self.days / 365.0
        self._simulate_terminal_prices(seed=self._seed)
        return (p_next - base) / float(days_eps)

    def greeks(self, option_type='call'):
        g = self.greeks_with_errors(option_type, use_pathwise=True)
        return {
            'delta': g['delta'][0],
            'gamma': g['gamma'][0],
            'vega': g['vega'][0],
            'theta': g['theta'][0],
            'rho': g['rho'][0],
        }

    # -----------------------
    # Plotting
    # -----------------------
    def plot_simulation_results(self, num_of_movements=20, seed=None):
        num_of_movements = int(max(1, min(num_of_movements, 200)))
        rng = np.random.default_rng(seed if seed is not None else self._seed)
        num_steps = max(1, self.days)
        dt = self.T / float(num_steps) if num_steps > 0 else 0.0
        paths = np.zeros((num_steps + 1, num_of_movements))
        paths[0, :] = self.S_0
        for t in range(1, num_steps + 1):
            Z = rng.standard_normal(num_of_movements)
            paths[t, :] = paths[t - 1, :] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z)
        fig = plt.figure(figsize=(10, 6))
        plt.plot(paths)
        plt.axhline(self.K, linestyle='--', linewidth=1.0, label='Strike Price')
        plt.xlim([0, paths.shape[0] - 1])
        plt.ylabel('Simulated price movements')
        plt.xlabel('Steps')
        plt.title(f'Sample {num_of_movements} simulated paths ({self.variance_reduction} variance reduction)')
        plt.legend(loc='best')
        plt.grid(True)
        return fig
