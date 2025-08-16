# notebooks/case_study.py
"""
Case study script that runs a few representative analyses and writes a short PDF report.

Produces:
 - figs/bs_vs_fft.png
 - figs/mc_greeks.png
 - report/case_study.pdf
"""

import os
os.makedirs("figs", exist_ok=True)
os.makedirs("report", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from option_pricing import BlackScholesModel, MonteCarloPricing

# 1) BS price example
S, K, days, r, sigma = 100.0, 100.0, 90, 0.01, 0.2
bsm = BlackScholesModel(S, K, days, r, sigma)
call = bsm.calculate_option_price("Call Option")
put = bsm.calculate_option_price("Put Option")

# plot price vs strike (simple)
strikes = np.linspace(60,140,41)
prices = [BlackScholesModel(S, k, days, r, sigma).calculate_option_price("Call Option") for k in strikes]
plt.figure(figsize=(8,4))
plt.plot(strikes, prices, label="BS call price")
plt.axvline(K, color='k', linestyle='--', label=f'K={K}')
plt.title("Black–Scholes call price vs strike")
plt.xlabel("Strike"); plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figs/bs_vs_strike.png", dpi=150)
plt.close()

# 2) MC Greeks sample
mc = MonteCarloPricing(S, K, days, r, sigma, number_of_simulations=5000, variance_reduction='antithetic')
mc.simulate_prices()
g_se = mc.greeks_with_errors('call', use_pathwise=True)
# build a small bar plot
labels = ['Delta','Vega']
est = [g_se['delta'][0], g_se['vega'][0]]
errs = [g_se['delta'][1], g_se['vega'][1]]
plt.figure(figsize=(6,4))
plt.bar(labels, est, yerr=errs, capsize=5)
plt.title("MC Greeks (pathwise) estimates")
plt.savefig("figs/mc_greeks.png", dpi=150)
plt.close()

# Build PDF report
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    doc = SimpleDocTemplate("report/case_study.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Option Pricing Project — Mini Case Study", styles['Title']))
    story.append(Spacer(1,12))
    story.append(Paragraph(f"Black–Scholes sample: S={S}, K={K}, days={days}, r={r}, sigma={sigma}", styles['Normal']))
    story.append(Image("figs/bs_vs_strike.png", width=480, height=200))
    story.append(Spacer(1,12))
    story.append(Paragraph("Monte Carlo Greeks (pathwise) sample", styles['Heading2']))
    story.append(Image("figs/mc_greeks.png", width=360, height=180))
    doc.build(story)
    print("Report written to report/case_study.pdf")
except Exception as ex:
    print("Could not write PDF (reportlab missing?):", ex)
    print("But figures saved to figs/")
