#option-pricing-models/tests/test_lsm_sanity.py
import numpy as np
from option_pricing import LongstaffSchwartz
import pytest

def test_lsm_price_between_intrinsic_and_spot():
    S = 100.0
    K = 100.0
    r = 0.01
    sigma = 0.2
    days = 90
    # short test paths and low degree for speed
    lsm = LongstaffSchwartz(S, K, r, sigma, days, n_paths=2000, n_steps=30, payoff='put', poly_degree=2, seed=42)
    price = lsm.price()
    # American put price must be >= intrinsic (max(K-S,0)) and <= S (loose upper bound)
    intrinsic = max(K - S, 0.0)
    assert price >= intrinsic - 1e-8
    assert price < S * 2  # sanity upper bound
