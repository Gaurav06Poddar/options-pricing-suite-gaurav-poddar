# 📈 Option Pricing Models – Quantitative Finance Toolkit

Developed by **[Gaurav Poddar](https://www.linkedin.com/in/gauravpoddar-gp13/)**  
📍 Based in New York City | 🎓 Computer Science @ [New York University](https://www.nyu.edu)  
✉️ Email: gp2610@nyu.edu | gp2610@stern.nyu.edu  
💻 GitHub: [Gaurav06Poddar](https://github.com/Gaurav06Poddar)

---

## 👋 About Me

Hi, I’m **Gaurav Poddar**, a Computer Science student at NYU with a strong focus on **quantitative finance, AI/ML, and computational modeling**.  

Living in the heart of NYC, I’ve immersed myself in both **engineering and finance** — blending deep technical skills with market intuition. My passion lies in **building robust quant tools** that combine mathematical precision with practical usability.  

This project is my **quant playground** — a full-featured **option pricing lab** designed to showcase not only theory, but also **production-grade code** that recruiters and hiring managers at quant firms/banks would expect from a strong candidate.  

---

## 🚀 Project Overview

This app is an **interactive Streamlit-based platform** for **option pricing, risk analysis, volatility modeling, and hedging strategies**.  

It goes **beyond textbook implementations** — I designed it to be:  
- **Educational** (demonstrates the math behind models).  
- **Practical** (pulls **real market data** via Yahoo Finance).  
- **Research-Ready** (includes advanced models like Heston & Dupire).  

---

## 🧮 Features

### ✅ Core Models
- **Black–Scholes** closed-form pricing + Greeks (Delta, Gamma, Vega, Theta, Rho).
- **Binomial Tree** pricing (European & American) with **early exercise** support.
- **Monte Carlo** simulation (European) with:
  - Variance reduction (antithetic, control variates).
  - Low-variance Greeks (pathwise / likelihood ratio).
  - Standard error estimates.

### ✅ Advanced Pricing
- **Longstaff–Schwartz Monte Carlo** for American options.
- **Asian Options** pricing with control variates.
- **Carr–Madan FFT pricer** (demo with Black–Scholes CF).
- **Dupire local volatility surface** estimation.

### ✅ Volatility Lab
- Realized vs Implied volatility analysis.
- **Delta-hedged vol-arbitrage backtests**.
- Execution cost modeling (fixed, linear slippage, quadratic impact).
- Heatmaps & grid-search for threshold/rebalancing strategies.

### ✅ Market Data & IV Tools
- **Live data integration** via Yahoo Finance (`yfinance`).
- Extract **real market IVs** from option chains.
- Build **synthetic implied volatility surfaces**.
- Heston model calibration (single expiry).
- Semi-closed form **Heston pricing**.

### 💹 Risk Dashboard
- **Portfolio risk metrics**: VaR (Value-at-Risk), CVaR (Expected Shortfall).
- **Stress testing**: user-defined shocks to underlying price/volatility.
- **Greeks aggregation**: portfolio-level Delta/Gamma/Vega exposures.

### 💹 Paper Trading & Performance Tracking
- **Paper Trading Engine**:
   Place simulated trades on **live-market option chains** (via Yahoo Finance).
   Track positions, P&L, realized/unrealized gains.
- **Performance Page**:
   Daily equity curve visualization.
   Trade logs, win/loss ratio, drawdowns.
   **Sharpe ratio**, volatility-adjusted returns.

---

## 🎨 Demo Previews

The app includes **interactive 3D surfaces, volatility term structures, and heatmaps** — making it not just functional, but visually compelling for recruiter demos.

---

## 🛠️ Tech Stack

- **Python** (NumPy, SciPy, Pandas, Matplotlib, Plotly)  
- **Streamlit** (for interactive dashboard)  
- **yfinance** (for real-time market data)  
- **pytest** (unit testing of models)  

---

## ⚡ Installation & Usage

Clone the repo:

```bash
git clone https://github.com/Gaurav06Poddar/option-pricing-suite-gaurav-poddar.git
cd option-pricing-models
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```