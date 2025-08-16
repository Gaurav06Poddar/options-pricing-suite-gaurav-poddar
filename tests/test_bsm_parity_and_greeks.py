#option-pricing-models/tests/test_bsm_parity_and_greeks.py
import math
import numpy as np
from option_pricing import BlackScholesModel
import pytest

def test_put_call_parity_and_positive():
    S = 100.0
    K = 100.0
    days = 180
    r = 0.01
    sigma = 0.2
    bsm = BlackScholesModel(S, K, days, r, sigma)
    c = bsm.calculate_option_price('Call Option')
    p = bsm.calculate_option_price('Put Option')
    # put-call parity: C - P = S - K * exp(-rT)
    T = days / 365.0
    lhs = c - p
    rhs = S - K * math.exp(-r * T)
    assert pytest.approx(lhs, rel=1e-6, abs=1e-6) == rhs
    assert c >= 0.0
    assert p >= 0.0

def test_greeks_reasonable():
    S = 120.0
    K = 100.0
    days = 30
    r = 0.01
    sigma = 0.25
    bsm = BlackScholesModel(S, K, days, r, sigma)
    g = bsm.greeks('call')
    # delta in (0,1), vega positive
    assert 0.0 < g['delta'] < 1.0
    assert g['vega'] > 0.0
