#option-pricing-models/tests/test_carr_madan_vs_bs.py
import numpy as np
import pytest
from option_pricing import BlackScholesModel
# FFT module optional
try:
    from option_pricing.carr_madan import cf_bs, carr_madan_fft
    _FFT_OK = True
except Exception:
    _FFT_OK = False

@pytest.mark.skipif(not _FFT_OK, reason="Carr–Madan not available")
def test_fft_vs_bs_sample():
    S0 = 100.0; r = 0.01; sigma = 0.2; T = 0.5
    cf = lambda u: cf_bs(u, S0, r, sigma, T)
    strikes, calls = carr_madan_fft(cf, S0=S0, r=r, T=T, alpha=1.5, eta=0.25, N=2**9, B=60.0)
    # select mid-range strike
    K = strikes[len(strikes)//2]
    bs = BlackScholesModel(S0, K, int(T*365), r, sigma).calculate_option_price('Call Option')
    # find FFT price for strike K
    idx = int(np.argmin(np.abs(strikes - K)))
    fft_price = calls[idx]
    assert abs(fft_price - bs) / max(bs, 1e-6) < 0.02  # within 2%
