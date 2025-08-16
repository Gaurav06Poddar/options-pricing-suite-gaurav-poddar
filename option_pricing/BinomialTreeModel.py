# Third party imports
import numpy as np

# Local package imports
from .base import OptionPricingModel


class BinomialTreeModel(OptionPricingModel):
    """
    Binomial tree European/American option pricing + finite-difference Greeks.
    Set american=True to enable early-exercise during rollback.
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity,
                 risk_free_rate, sigma, number_of_time_steps=1000, american=False):
        self.S = float(underlying_spot_price)
        self.K = float(strike_price)
        self.days = int(days_to_maturity)
        self.T = float(days_to_maturity) / 365.0
        self.r = float(risk_free_rate)
        self.sigma = float(sigma)
        self.number_of_time_steps = int(number_of_time_steps)
        self.american = bool(american)

    # -------------------------
    # Pricing
    # -------------------------
    def _price(self, option_type='call'):
        """Internal helper to compute price for given option_type."""
        # handle immediate expiry
        if self.T <= 0 or self.number_of_time_steps <= 0:
            intrinsic = (self.S - self.K) if option_type == 'call' else (self.K - self.S)
            return max(0.0, intrinsic)

        dT = self.T / self.number_of_time_steps
        u = np.exp(self.sigma * np.sqrt(dT))
        d = 1.0 / u
        a = np.exp(self.r * dT)
        p = (a - d) / (u - d)
        q = 1.0 - p

        # Prices at maturity
        S_T = np.array([self.S * (u ** j) * (d ** (self.number_of_time_steps - j))
                        for j in range(self.number_of_time_steps + 1)])
        if option_type == 'call':
            V = np.maximum(S_T - self.K, 0.0)
        else:
            V = np.maximum(self.K - S_T, 0.0)

        # Backward induction
        for i in range(self.number_of_time_steps - 1, -1, -1):
            V = np.exp(-self.r * dT) * (p * V[1:] + q * V[:-1])
            if self.american:
                S_nodes = np.array([self.S * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
                if option_type == 'call':
                    intrinsic = np.maximum(S_nodes - self.K, 0.0)
                else:
                    intrinsic = np.maximum(self.K - S_nodes, 0.0)
                V = np.maximum(V, intrinsic)

        return float(V[0])

    def _calculate_call_option_price(self):
        return self._price('call')

    def _calculate_put_option_price(self):
        return self._price('put')

    # -------------------------
    # Greeks (finite differences)
    # -------------------------
    def _finite_difference(self, param, bump, option_type='call', central=True):
        """
        Generic finite-difference helper.
        param: 'S', 'sigma', 'r', 'days'
        bump: absolute bump amount
        central: whether to use central difference
        """
        orig = getattr(self, param)

        if central:
            setattr(self, param, orig + bump)
            if param == 'days':
                self.T = float(self.days) / 365.0
            price_plus = self._price(option_type)

            setattr(self, param, orig - bump)
            if param == 'days':
                self.T = float(self.days) / 365.0
            price_minus = self._price(option_type)

            setattr(self, param, orig)
            if param == 'days':
                self.T = float(self.days) / 365.0

            return 0.5 * (price_plus - price_minus) / bump
        else:
            setattr(self, param, orig + bump)
            if param == 'days':
                self.T = float(self.days) / 365.0
            price_plus = self._price(option_type)

            setattr(self, param, orig)
            if param == 'days':
                self.T = float(self.days) / 365.0
            base = self._price(option_type)
            return (price_plus - base) / bump

    def delta(self, option_type='call', eps=1e-2):
        return self._finite_difference('S', eps, option_type, central=True)

    def gamma(self, option_type='call', eps=1e-2):
        origS = self.S
        self.S = origS + eps
        price_plus = self._price(option_type)
        self.S = origS
        price = self._price(option_type)
        self.S = origS - eps
        price_minus = self._price(option_type)
        self.S = origS
        return (price_plus - 2 * price + price_minus) / (eps ** 2)

    def vega(self, option_type='call', eps=1e-4):
        return self._finite_difference('sigma', eps, option_type, central=True)

    def rho(self, option_type='call', eps=1e-4):
        return self._finite_difference('r', eps, option_type, central=True)

    def theta(self, option_type='call', days_eps=1):
        if self.days - days_eps <= 0:
            return 0.0
        orig_days = self.days
        orig_T = self.T

        price_today = self._price(option_type)
        self.days = orig_days - days_eps
        self.T = float(self.days) / 365.0
        price_next = self._price(option_type)

        self.days = orig_days
        self.T = orig_T
        return (price_next - price_today) / float(days_eps)

    def greeks(self, option_type='call', eps_S=1e-2, eps_vol=1e-4, eps_r=1e-4, days_eps=1):
        return {
            "delta": self.delta(option_type, eps=eps_S),
            "gamma": self.gamma(option_type, eps=eps_S),
            "vega": self.vega(option_type, eps=eps_vol),
            "theta": self.theta(option_type, days_eps=days_eps),
            "rho": self.rho(option_type, eps=eps_r),
        }
