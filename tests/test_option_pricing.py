#option-pricing-models/tests/test_option_pricing.py
import math
import numpy as np
import pytest

from option_pricing import BlackScholesModel, BinomialTreeModel, MonteCarloPricing, LongstaffSchwartz

# ---- Black Scholes analytic tests ----
@pytest.mark.parametrize("S,K,days,r,sigma", [
    (100.0, 100.0, 365, 0.01, 0.2),
    (120.0, 100.0, 180, 0.02, 0.3),
    (50.0, 55.0, 90, 0.005, 0.25),
])
def test_black_scholes_price_sign_and_greeks(S, K, days, r, sigma):
    bsm = BlackScholesModel(S, K, days, r, sigma)
    c = bsm.calculate_option_price('Call Option')
    p = bsm.calculate_option_price('Put Option')

    # basic sanity: non-negative prices
    assert c >= 0.0
    assert p >= 0.0

    greeks_call = bsm.greeks('call')
    # Delta should be between 0 and 1 for call, gamma non-negative, vega non-negative
    assert 0.0 <= greeks_call['delta'] <= 1.0
    assert greeks_call['gamma'] >= 0.0
    assert greeks_call['vega'] >= 0.0

    # Put-call parity approximate: C - P ≈ S - K e^{-rT}
    lhs = c - p
    rhs = S - K * math.exp(-r * (days / 365.0))
    assert pytest.approx(lhs, rel=1e-6, abs=1e-6) == pytest.approx(rhs, rel=1e-6, abs=1e-6)

# ---- LSM sanity checks ----
def test_lsm_put_between_intrinsic_and_binomial():
    S = 100.0
    K = 100.0
    days = 90
    r = 0.01
    sigma = 0.2

    # Binomial american with many steps as "high-quality" reference
    binom_high = BinomialTreeModel(S, K, days, r, sigma, number_of_time_steps=2000, american=True)
    binom_price = binom_high.calculate_option_price('Put Option')

    # LSM
    lsm = LongstaffSchwartz(S, K, r, sigma, days, n_paths=10000, n_steps=90, payoff='put', poly_degree=2, seed=1234)
    lsm_price = lsm.price()

    # intrinsic at t=0 is max(K-S,0) = 0
    intrinsic = max(K - S, 0.0)

    # LSM price should be >= intrinsic and approximately <= high-quality binomial
    assert lsm_price >= intrinsic - 1e-8
    # allow some tolerance: LSM may be slightly below binomial if regression imperfect
    assert lsm_price <= binom_price + 0.5  # 0.5 $ tolerance acceptable for moderate paths

# ---- Monte Carlo pathwise Greek sanity ----
def test_mc_pathwise_delta_consistent_with_bsm():
    S = 100.0; K = 100.0; days = 30; r = 0.01; sigma = 0.2
    # reference BSM delta
    bsm = BlackScholesModel(S, K, days, r, sigma)
    bsm_delta = bsm.delta('call')

    mc = MonteCarloPricing(S, K, days, r, sigma, number_of_simulations=20000, variance_reduction='antithetic')
    mc.simulate_prices(seed=12345)
    # compute PW delta (function exposed by MC implementation)
    try:
        pw_delta = mc.pathwise_delta(option_type='call')
    except AttributeError:
        pytest.skip("pathwise_delta not implemented")
    # should be close in expectation
    assert abs(pw_delta - bsm_delta) < 0.05  # 5 cent tolerance on delta for 20k sims

