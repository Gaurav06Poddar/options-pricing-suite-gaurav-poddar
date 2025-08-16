#option-pricing-models/option_pricing_benchmark.py
"""
Benchmarking & comparison script for Phase 1.

Run:
    python option_pricing_benchmark.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from option_pricing import BlackScholesModel, BinomialTreeModel, MonteCarloPricing

# ---------- Parameters ----------
S0 = 100.0
K = 100.0
days = 365
r = 0.01
sigma = 0.2

# Varying parameters to study trade-offs
binomial_steps_list = [50, 200, 1000, 5000]
mc_sims_list = [1000, 5000, 10000, 50000]

# store results
binomial_results = []
mc_results = []

# Reference Black-Scholes
bsm = BlackScholesModel(S0, K, days, r, sigma)
bsm_price_call = bsm.calculate_option_price('Call Option')
bsm_greeks_call = bsm.greeks('call')
print("Black-Scholes Call Price (reference):", bsm_price_call)
print("Black-Scholes Greeks:", bsm_greeks_call)

# ---------- Binomial benchmarks ----------
for steps in binomial_steps_list:
    start = time.perf_counter()
    model = BinomialTreeModel(S0, K, days, r, sigma, number_of_time_steps=steps)
    price = model.calculate_option_price('Call Option')
    runtime = time.perf_counter() - start
    greeks = model.greeks('call')
    abs_err = abs(price - bsm_price_call)
    rel_err = abs_err / (abs(bsm_price_call) + 1e-12)
    binomial_results.append({
        'steps': steps,
        'price': price,
        'runtime': runtime,
        'abs_err': abs_err,
        'rel_err': rel_err,
        'greeks': greeks
    })
    print(f"[Binomial steps={steps}] price={price:.6f}, abs_err={abs_err:.6f}, runtime={runtime:.4f}s")

# ---------- Monte Carlo benchmarks ----------
for sims in mc_sims_list:
    start = time.perf_counter()
    mc = MonteCarloPricing(S0, K, days, r, sigma, number_of_simulations=sims)
    mc.simulate_prices()  # ensures CRN seed is set
    price = mc.calculate_option_price('Call Option')
    runtime = time.perf_counter() - start
    greeks = mc.greeks('call')
    abs_err = abs(price - bsm_price_call)
    rel_err = abs_err / (abs(bsm_price_call) + 1e-12)
    mc_results.append({
        'sims': sims,
        'price': price,
        'runtime': runtime,
        'abs_err': abs_err,
        'rel_err': rel_err,
        'greeks': greeks
    })
    print(f"[MC sims={sims}] price={price:.6f}, abs_err={abs_err:.6f}, runtime={runtime:.4f}s")

# ---------- Plotting ----------
# Binomial: error vs runtime
plt.figure(figsize=(8, 5))
plt.plot([r['runtime'] for r in binomial_results], [r['abs_err'] for r in binomial_results], marker='o', label='Binomial')
for r in binomial_results:
    plt.text(r['runtime'], r['abs_err'], f"n={r['steps']}")
# MC: error vs runtime
plt.plot([r['runtime'] for r in mc_results], [r['abs_err'] for r in mc_results], marker='x', label='Monte Carlo')
for r in mc_results:
    plt.text(r['runtime'], r['abs_err'], f"N={r['sims']}")
plt.xlabel('Runtime (s)')
plt.ylabel('Absolute price error vs Black-Scholes')
plt.title('Runtime vs Absolute Error (Call option)')
plt.legend()
plt.grid(True)
plt.show()

# ---------- Summary print ----------
print("\n--- Summary ---\n")
print("Binomial results:")
for r in binomial_results:
    print(r)
print("\nMonte Carlo results:")
for r in mc_results:
    print(r)
