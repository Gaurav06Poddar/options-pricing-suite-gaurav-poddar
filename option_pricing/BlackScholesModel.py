# Third party imports
import numpy as np
from scipy.stats import norm

# Local package imports
from .base import OptionPricingModel


class BlackScholesModel(OptionPricingModel):
    """
    Black-Scholes closed-form pricing + Greeks for European options (no dividends).
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma):
        self.S = float(underlying_spot_price)
        self.K = float(strike_price)
        self.days = float(days_to_maturity)
        self.T = float(days_to_maturity) / 365.0
        self.r = float(risk_free_rate)
        self.sigma = float(sigma)

        # Precompute d1, d2 (guard for T==0)
        self._compute_d1_d2_and_price()

    def _compute_d1_d2_and_price(self):
        if self.T <= 0 or self.sigma <= 0:
            self.d1 = None
            self.d2 = None
            self.call_price = max(0.0, self.S - self.K)
            self.put_price = max(0.0, self.K - self.S)
            return

        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

        self.N_d1 = norm.cdf(self.d1)
        self.N_d2 = norm.cdf(self.d2)
        self.N_minus_d1 = norm.cdf(-self.d1)
        self.N_minus_d2 = norm.cdf(-self.d2)

        self.call_price = self.S * self.N_d1 - self.K * np.exp(-self.r * self.T) * self.N_d2
        self.put_price = self.K * np.exp(-self.r * self.T) * self.N_minus_d2 - self.S * self.N_minus_d1

    def _calculate_call_option_price(self):
        return self.call_price

    def _calculate_put_option_price(self):
        return self.put_price

    # -------------------------
    # Greeks (closed-form)
    # -------------------------
    def delta(self, option_type='call'):
        if self.T <= 0 or self.sigma <= 0:
            return 1.0 if (option_type == 'call' and self.S > self.K) else (0.0 if option_type == 'call' else (-1.0 if self.S < self.K else 0.0))
        if option_type == 'call':
            return norm.cdf(self.d1)
        else:
            return norm.cdf(self.d1) - 1.0

    def gamma(self):
        if self.T <= 0 or self.sigma <= 0:
            return 0.0
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        if self.T <= 0 or self.sigma <= 0:
            return 0.0
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T)

    def theta(self, option_type='call'):
        if self.T <= 0 or self.sigma <= 0:
            return 0.0
        pdf_d1 = norm.pdf(self.d1)
        first_term = - (self.S * pdf_d1 * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == 'call':
            second_term = - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            return first_term + second_term
        else:
            second_term = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            return first_term + second_term

    def rho(self, option_type='call'):
        if self.T <= 0 or self.sigma <= 0:
            return 0.0
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)

    def greeks(self, option_type='call'):
        opt = option_type.lower()
        return {
            "delta": self.delta(opt),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(opt),
            "rho": self.rho(opt),
        }

    # -------------------------
    # Implied volatility
    # -------------------------
    def implied_volatility(self, market_price, option_type='call', tol=1e-6, max_iter=200):
        """
        Solve for implied volatility via bisection (robust).
        market_price: observed option premium
        option_type: 'call' or 'put'
        Returns implied volatility (annual), or None if not found.
        """
        # trivial cases
        if market_price <= 0:
            return 0.0

        # objective function: BS price(sigma) - market_price
        def bs_price(sigma):
            if sigma <= 0:
                # intrinsic discounted maybe
                if option_type == 'call':
                    return max(0.0, self.S - self.K * np.exp(-self.r * self.T))
                else:
                    return max(0.0, self.K * np.exp(-self.r * self.T) - self.S)
            d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
            d2 = d1 - sigma * np.sqrt(self.T)
            if option_type == 'call':
                return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            else:
                return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

        # set search bounds for sigma
        low = 1e-12
        high = 5.0  # 500% vol upper bound (should be enough)
        f_low = bs_price(low) - market_price
        f_high = bs_price(high) - market_price

        if f_low * f_high > 0:
            # no sign change -> cannot find root within bounds
            return None

        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            f_mid = bs_price(mid) - market_price
            if abs(f_mid) < tol:
                return float(mid)
            if f_low * f_mid <= 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return float(0.5 * (low + high))
