#option-pricing-models/streamlit_app.py
import streamlit as st
from enum import Enum
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --- helpers to normalize IV surface frames (for the 3D plot) ---
def _normalize_surface_df(surf: pd.DataFrame, use_calls: bool = True) -> pd.DataFrame:
    """
    Make sure the returned surface has columns: 'strike', 'T', 'iv'.
    Tries to auto-rename common alternatives and flatten any index.
    """
    if not isinstance(surf, pd.DataFrame):
        surf = pd.DataFrame(surf)

    # Flatten index if needed
    if isinstance(surf.index, pd.MultiIndex):
        surf = surf.reset_index()
    else:
        # If the index is actually strike, expose it
        if surf.index.name and surf.index.name.lower() in {"strike", "k", "strike_price"}:
            surf = surf.reset_index()

    # Lowercase lookup
    cols_l = {c.lower(): c for c in surf.columns}

    # Map strike
    if "strike" not in surf.columns:
        for cand in ("strike", "k", "strike_price", "strikes", "x"):
            if cand in cols_l:
                surf = surf.rename(columns={cols_l[cand]: "strike"})
                break

    # Map T (years). Try common names; derive from days if needed
    if "T" not in surf.columns:
        found_T = None
        for cand in ("t", "tau", "maturity", "years", "time_to_expiry", "time", "ttm"):
            if cand in cols_l:
                surf = surf.rename(columns={cols_l[cand]: "T"})
                found_T = "T"
                break
        if not found_T:
            for cand in ("days_to_expiry", "days", "dte"):
                if cand in cols_l:
                    surf["T"] = np.asarray(surf[cols_l[cand]], dtype=float) / 365.0
                    found_T = "T"
                    break

    # Map/compute 'iv'
    if "iv" not in surf.columns:
        chosen = None
        # prefer the side the user selected
        side_col = "call_iv" if use_calls else "put_iv"
        if side_col in cols_l:
            chosen = cols_l[side_col]
        else:
            for cand in ("iv", "iv_mid", "implied_vol", "sigma_imp", "vol", "volatility"):
                if cand in cols_l:
                    chosen = cols_l[cand]
                    break
        if chosen is None:
            # combine call/put if both present: take first non-null
            if "call_iv" in cols_l or "put_iv" in cols_l:
                surf["iv"] = surf.get(cols_l.get("call_iv"))\
                              .fillna(surf.get(cols_l.get("put_iv")))
            else:
                # nothing to use -> let it fail gracefully later
                pass
        else:
            surf = surf.rename(columns={chosen: "iv"})

    # Final trim & dtype
    if "strike" in surf.columns:
        surf["strike"] = surf["strike"].astype(float)
    if "T" in surf.columns:
        surf["T"] = surf["T"].astype(float)
    if "iv" in surf.columns:
        surf["iv"] = surf["iv"].astype(float)

    return surf


# --- robust Heston fallback (Lewis/Heston P1/P2 formulation; pure NumPy trapz) ---
def _heston_call_fallback(S0, K, T, r, kappa, theta, sigma_v, rho, v0,
                          umax=200.0, N=4096):
    """
    Heston call via original P1/P2 integrals with simple trapezoidal rule.
    Used ONLY if the package's HestonModel fails/returns nan.
    """
    S0 = float(S0); K = float(K); T = max(float(T), 1e-8)
    r  = float(r)
    kappa = max(float(kappa), 1e-12)
    theta = max(float(theta), 1e-12)
    sigma_v = max(float(sigma_v), 1e-12)
    rho = float(np.clip(rho, -0.999, 0.999))
    v0  = max(float(v0), 0.0)

    x = np.log(S0)
    u = np.linspace(1e-6, umax, int(N))

    def _fj(u, j):
        iu = 1j*u
        a = kappa * theta
        b = kappa - rho*sigma_v if j == 1 else kappa
        d = np.sqrt((rho*sigma_v*iu - b)**2 + (sigma_v**2)*(iu + u**2))
        g = (b - rho*sigma_v*iu - d) / (b - rho*sigma_v*iu + d)
        e_dt = np.exp(-d*T)
        # avoid log of zero
        one_minus_g = (1 - g)
        one_minus_gedt = (1 - g*e_dt)
        # add tiny eps where needed to avoid log(0)
        one_minus_g = np.where(np.abs(one_minus_g) < 1e-15, one_minus_g + 1e-15, one_minus_g)
        one_minus_gedt = np.where(np.abs(one_minus_gedt) < 1e-15, one_minus_gedt + 1e-15, one_minus_gedt)

        C = (r*iu*T) + (a/(sigma_v**2)) * ((b - rho*sigma_v*iu - d)*T - 2.0*np.log(one_minus_gedt/one_minus_g))
        D = ((b - rho*sigma_v*iu - d)/(sigma_v**2)) * ((1 - e_dt)/one_minus_gedt)
        return np.exp(C + D*v0 + iu*x)

    def _P(j):
        fj = _fj(u, j)
        integrand = np.real(np.exp(-1j*u*np.log(K)) * fj / (1j*u))
        return 0.5 + (1/np.pi) * np.trapz(integrand, u)

    P1 = _P(1)
    P2 = _P(2)
    call = S0*P1 - K*np.exp(-r*T)*P2
    return float(call)

def _heston_put_from_call(call, S0, K, T, r):
    return float(call - S0 + K*np.exp(-float(r)*float(T)))


# Core models & utilities from the package (Phase 1..8)
from option_pricing import (
    BlackScholesModel,
    MonteCarloPricing,
    BinomialTreeModel,
    LongstaffSchwartz,
    AsianMonteCarlo,
    Ticker,
)

# Import Phase 3/4/5/6/7/8 helpers if present (guarded imports)
try:
    from option_pricing.market_iv_surface import (
        fetch_expiries,
        implied_vol_from_chain,
        build_surface,
    )
except Exception:
    # Some functions may be in different modules depending on your code layout
    fetch_expiries = None
    implied_vol_from_chain = None
    build_surface = None

try:
    from option_pricing.heston_calibration import (
        HestonParams,
        calibrate_heston_to_iv,
        heston_implied_vol_curve,
        summarize_fit,
    )
    from option_pricing.HestonModel import HestonModel
except Exception:
    HestonParams = None
    calibrate_heston_to_iv = None
    heston_implied_vol_curve = None
    summarize_fit = None
    HestonModel = None

# Vol / strategy / hedging / FFT / Dupire modules from later phases
try:
    from option_pricing.vol_analysis import (
        compute_rolling_realized_vs_implied,
        aggregate_implied_term_structure,
        fetch_expiries as vol_fetch_expiries,
        compute_realized_vs_implied_surface
    )
except Exception:
    # fallbacks
    compute_rolling_realized_vs_implied = None
    aggregate_implied_term_structure = None
    vol_fetch_expiries = None
    compute_realized_vs_implied_surface = None

try:
    from option_pricing.vol_strategy import vol_arbitrage_strategy_backtest, grid_search_vol_arb
except Exception:
    vol_arbitrage_strategy_backtest = None
    grid_search_vol_arb = None

try:
    from option_pricing.hedging import backtest_dynamic_hedge, simulate_black_scholes_paths
except Exception:
    backtest_dynamic_hedge = None
    simulate_black_scholes_paths = None

try:
    from option_pricing.carr_maden import cf_bs, carr_madan_fft
except Exception:
    cf_bs = None
    carr_madan_fft = None

try:
    from option_pricing.dupire import dupire_local_vol
except Exception:
    dupire_local_vol = None

try:
    from option_pricing.optimal_hedge import compute_mv_hedge_ratio_mc, compute_mv_hedge_ratio_historical
except Exception:
    compute_mv_hedge_ratio_mc = None
    compute_mv_hedge_ratio_historical = None

# Page layout
st.set_page_config(page_title="Option Pricing Suite", layout="wide", initial_sidebar_state="expanded")
st.title("Option Pricing Suite — Gaurav Poddar")

PAGES = [
    "European (BS / Binomial / MC)",
    "American (Binomial / LSM)",
    "Asian (Arithmetic MC)",
    "Implied Volatility (Synthetic Demo)",
    "Market IV (Real Data) & Heston",
    "Volatility Lab (Vol-Arb & Grid Search)",
    "Advanced: FFT / LocalVol / Hedging",
    "Paper Trading (Runner)",
    "Risk Dashboard",
    "Performance",
    "About & Notes",
]
page = st.sidebar.radio("App section", PAGES)

# -----------------------
# Helpers & caching
# -----------------------
@st.cache_data
def get_current_price(ticker: str):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return float(data['Close'].iloc[-1])
    except Exception:
        return None

@st.cache_data
def get_historical_data(ticker: str):
    try:
        return Ticker.get_historical_data(ticker)
    except Exception:
        return None

def format_greeks(gdict):
    vega_per_pct = (gdict.get('vega', 0.0) or 0.0) / 100.0
    theta_per_day = (
        (gdict.get('theta', 0.0) or 0.0) / 365.0
        if ('theta' in gdict and gdict.get('theta') is not None)
        else None
    )
    return {
        'Delta': gdict.get('delta', 0.0),
        'Gamma': gdict.get('gamma', 0.0),
        'Vega (per unit vol)': gdict.get('vega', 0.0),
        'Vega (per 1%)': vega_per_pct,
        'Theta (per day)': theta_per_day,
        'Rho (per unit rate)': gdict.get('rho', 0.0),
    }

# =========================
# European (BS / Binomial / MC)
# =========================
if page == "European (BS / Binomial / MC)":
    st.header("European Options — Pricing & Greeks")
    col1, col2 = st.columns([1, 1])
    with col1:
        ticker = st.text_input("Ticker (for historical)", value="AAPL")
        data = get_historical_data(ticker)
        price_hint = get_current_price(ticker)
        if price_hint:
            st.write("Current price:", f"${price_hint:.2f}")
        else:
            st.info("Unable to fetch current price; provide spot manually below.")

    with col2:
        spot = st.number_input("Spot price (overrides ticker)", value=float(price_hint) if price_hint is not None else 100.0, format="%.4f")
        strike = st.number_input("Strike price", value=round(spot, 0), step=0.5)
        days = st.number_input("Days to maturity", min_value=1, value=90)
        r_pct = st.number_input("Risk-free rate (%)", value=1.0, step=0.1)
        vol_pct = st.number_input("Volatility (sigma %) (annualized)", value=20.0, step=0.1)

    st.markdown("---")
    st.subheader("Black-Scholes (European)")
    colbs1, colbs2 = st.columns([1, 1])
    with colbs1:
        bsm = BlackScholesModel(spot, strike, int(days), r_pct / 100.0, vol_pct / 100.0)
        call_price = bsm.calculate_option_price("Call Option")
        put_price = bsm.calculate_option_price("Put Option")
        st.metric("BS Call Price", f"{call_price:.4f}")
        st.metric("BS Put Price", f"{put_price:.4f}")
    with colbs2:
        st.markdown("**Black-Scholes Greeks (call)**")
        st.table(pd.DataFrame([format_greeks(bsm.greeks('call'))]).T.rename(columns={0:"Value"}))

    st.markdown("---")
    st.subheader("Binomial (European)")
    colb1, colb2 = st.columns([1, 1])
    with colb1:
        steps = st.slider("Binomial steps", min_value=10, max_value=20000, value=1000, step=10)
        binom = BinomialTreeModel(spot, strike, int(days), r_pct / 100.0, vol_pct / 100.0, number_of_time_steps=steps, american=False)
        call_price_bin = binom.calculate_option_price("Call Option")
        put_price_bin = binom.calculate_option_price("Put Option")
        st.metric("Binomial Call Price", f"{call_price_bin:.4f}")
        st.metric("Binomial Put Price", f"{put_price_bin:.4f}")
    with colb2:
        st.markdown("**Binomial Greeks (call)**")
        st.table(pd.DataFrame([format_greeks(binom.greeks('call'))]).T.rename(columns={0:"Value"}))

    st.markdown("---")
    st.subheader("Monte Carlo (European) — Variance Reduction + Low-Variance Greeks")
    colm1, colm2 = st.columns([1,1])
    with colm1:
        sims = st.number_input("MC simulations", min_value=100, max_value=500000, value=10000, step=100)
        vr = st.selectbox("Variance reduction", ["None", "Antithetic", "Control", "Both"])
        vr_key = vr.lower()
        mc = MonteCarloPricing(spot, strike, int(days), r_pct / 100.0, vol_pct / 100.0, number_of_simulations=int(sims), variance_reduction=vr_key)
        mc.simulate_prices()
        call_mc = mc.calculate_option_price("Call Option")
        put_mc = mc.calculate_option_price("Put Option")
        st.metric("MC Call Price", f"{call_mc:.4f}")
        st.metric("MC Put Price", f"{put_mc:.4f}")

    with colm2:
        st.markdown("**MC Greeks (call)**")
        # show both scalar and SE-based
        g_scalar = mc.greeks('call')
        base_tbl = pd.DataFrame([format_greeks(g_scalar)]).T.rename(columns={0: "Estimate"})
        st.table(base_tbl)

        show_se = st.checkbox("Show SE-based estimators (pathwise/LR + CRN Gamma)", value=True)
        if show_se:
            gres = mc.greeks_with_errors('call', use_pathwise=True)
            df = pd.DataFrame({
                'Greek': ['Price','Delta','Gamma','Vega','Theta','Rho'],
                'Estimate': [gres['price'][0], gres['delta'][0], gres['gamma'][0], gres['vega'][0], gres['theta'][0], gres['rho'][0]],
                'StdErr':   [gres['price'][1], gres['delta'][1], gres['gamma'][1], gres['vega'][1], gres['theta'][1], gres['rho'][1]]
            }).set_index('Greek')
            st.caption("Low-variance estimators with estimated standard errors")
            st.table(df)
            st.download_button("Download Greeks (with SE) CSV", df.to_csv().encode('utf-8'),
                               file_name="mc_greeks_with_se.csv", mime="text/csv")

        # show sample paths
        nplot = st.slider("Paths to plot (small)", 0, min(200, int(sims/10) if sims>=10 else 10), 20)
        if nplot > 0:
            fig = mc.plot_simulation_results(nplot)
            st.pyplot(fig)

# =========================
# American (Binomial / LSM)
# =========================
elif page == "American (Binomial / LSM)":
    st.header("American Options — Binomial and Longstaff–Schwartz")
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker (for hint)", value="AAPL")
        hint_price = get_current_price(ticker)
        if hint_price:
            st.write("Current price:", f"${hint_price:.2f}")
    with col2:
        spot = st.number_input("Spot price", value=float(hint_price) if hint_price is not None else 100.0)
        strike = st.number_input("Strike price", value=round(spot, 0))
        days = st.number_input("Days to maturity", min_value=1, value=90)
        r_pct = st.number_input("Risk-free rate (%)", value=1.0)
        vol_pct = st.number_input("Volatility (sigma %) (annual)", value=25.0)
        payoff = st.selectbox("Payoff", ["put", "call"])

    st.markdown("---")
    st.subheader("Binomial (American)")
    steps = st.slider("Binomial steps (American)", min_value=10, max_value=20000, value=1000, step=10)
    binom_am = BinomialTreeModel(spot, strike, int(days), r_pct / 100.0, vol_pct / 100.0, number_of_time_steps=steps, american=True)
    st.write("Pricing (binomial, early exercise allowed)...")
    call_price = binom_am.calculate_option_price("Call Option") if payoff == 'call' else None
    put_price = binom_am.calculate_option_price("Put Option") if payoff == 'put' else None
    if payoff == 'call':
        st.metric("American Call Price (Binomial)", f"{call_price:.4f}")
        st.table(pd.DataFrame([format_greeks(binom_am.greeks('call'))]).T.rename(columns={0:"Value"}))
    else:
        st.metric("American Put Price (Binomial)", f"{put_price:.4f}")
        st.table(pd.DataFrame([format_greeks(binom_am.greeks('put'))]).T.rename(columns={0:"Value"}))

    st.markdown("---")
    st.subheader("Longstaff–Schwartz (LSM Monte Carlo) — American")
    n_paths = st.number_input("LSM MC paths", min_value=1000, max_value=200000, value=20000, step=1000)
    n_steps = st.number_input("LSM MC steps", min_value=5, max_value=365, value=min(90, int(days)), step=1)
    poly_deg = st.slider("Polynomial degree for regression", min_value=1, max_value=5, value=2)
    st.write("Running LSM Monte Carlo (this may take a moment for many paths)...")
    lsm_engine = LongstaffSchwartz(spot, strike, r_pct / 100.0, vol_pct / 100.0, int(days), n_paths=int(n_paths), n_steps=int(n_steps), payoff=payoff, poly_degree=poly_deg, seed=12345)
    lsm_price = lsm_engine.price()
    st.metric("LSM American price", f"{lsm_price:.4f}")

# =========================
# Asian (Arithmetic MC)
# =========================
elif page == "Asian (Arithmetic MC)":
    st.header("Asian Options — Arithmetic-average Monte Carlo")
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker", value="AAPL")
        hint_price = get_current_price(ticker)
    with col2:
        spot = st.number_input("Spot price", value=float(hint_price) if hint_price is not None else 100.0)
        strike = st.number_input("Strike", value=round(spot, 0))
        days = st.number_input("Days to maturity", min_value=1, value=90)
        r_pct = st.number_input("Risk-free rate (%)", value=1.0)
        vol_pct = st.number_input("Volatility (sigma %) (annual)", value=30.0)
        payoff = st.selectbox("Payoff", ["call", "put"])
    st.markdown("---")
    n_paths = st.number_input("MC paths", min_value=1000, max_value=200000, value=20000, step=1000)
    n_steps = st.number_input("Sampling steps (per path)", min_value=1, max_value=365, value=min(90, int(days)))
    vr = st.selectbox("Variance reduction", ["none", "control"])
    engine = AsianMonteCarlo(spot, strike, r_pct / 100.0, vol_pct / 100.0, int(days), n_paths=int(n_paths), n_steps=int(n_steps), payoff=payoff, variance_reduction=vr, seed=1234)
    st.write("Computing Asian MC price...")
    asian_price = engine.price()
    st.metric("Asian option (arithmetic) price", f"{asian_price:.4f}")

# =========================
# Implied Volatility (Synthetic Demo)
# =========================
elif page == "Implied Volatility (Synthetic Demo)":
    st.header("Implied Volatility Calculator & Surface (synthetic demo)")
    col1, col2 = st.columns(2)
    with col1:
        spot = st.number_input("Spot", value=100.0)
        strike = st.number_input("Strike", value=100.0)
        days = st.number_input("Days to maturity", min_value=1, value=30)
        r_pct = st.number_input("Risk-free rate (%)", value=1.0)
        vol_init = st.number_input("Vol initial guess (%)", value=20.0)
        market_price = st.number_input("Market option price ($)", value=2.0)
        opt_type = st.selectbox("Option type", ["call", "put"])
    with col2:
        if st.button("Compute implied volatility"):
            bsm = BlackScholesModel(spot, strike, int(days), r_pct/100.0, vol_init/100.0)
            iv = bsm.implied_volatility(market_price, option_type=opt_type)
            if iv is None:
                st.error("Implied vol could not be found in search bounds (0..500%). Try different price/parameters.")
            else:
                st.success(f"Implied volatility: {iv*100:.4f}% (annualized)")

    st.markdown("---")
    st.subheader("Implied Vol Surface Demo (synthetic)")
    st.write("Create a small grid and compute BSM implied vol from synthetic prices.")
    K_min = st.number_input("Min strike", value=80.0)
    K_max = st.number_input("Max strike", value=120.0)
    nK = st.number_input("Number of strikes", min_value=3, max_value=25, value=7)
    min_days = st.number_input("Min days", min_value=1, max_value=365, value=30)
    max_days = st.number_input("Max days", min_value=1, max_value=365, value=365)
    nT = st.number_input("Number of maturities", min_value=1, max_value=12, value=4)

    if st.button("Build synthetic surface"):
        strikes = np.linspace(K_min, K_max, int(nK))
        t_days = np.linspace(min_days, max_days, int(nT)).astype(int)
        surf = []
        for d in t_days:
            row = []
            for K in strikes:
                vol_true = 0.2 + 0.2 * abs(K - spot) / spot  # skew
                bsm = BlackScholesModel(spot, K, int(d), r_pct/100.0, vol_true)
                price = bsm.calculate_option_price("Call Option")
                bsm_probe = BlackScholesModel(spot, K, int(d), r_pct/100.0, 0.2)
                iv = bsm_probe.implied_volatility(price, option_type='call')
                row.append(iv if iv is not None else np.nan)
            surf.append(row)
        surf = np.array(surf)
        df = pd.DataFrame(surf, index=t_days, columns=np.round(strikes, 2))
        st.write("Implied vol surface (rows = days, cols = strikes):")
        st.dataframe(df.style.format("{:.4f}"))

# =========================
# Market IV (Real Data) & Heston
# =========================
elif page == "Market IV (Real Data) & Heston":
    st.header("Market Implied Vol (Real Data) + Heston Calibration")

    tcol = st.columns(3)
    ticker = tcol[0].text_input("Ticker", "AAPL")
    r_pct = tcol[1].number_input("Risk-free rate %", 0.0, 100.0, 2.0, step=0.1)
    use_calls = tcol[2].selectbox("Use", ["Calls", "Puts"], index=0) == "Calls"

    # Expiries selector
    if st.button("Fetch expiries"):
        try:
            exps = []
            if fetch_expiries is not None:
                exps = list(fetch_expiries(ticker))
            elif vol_fetch_expiries is not None:
                exps = list(vol_fetch_expiries(ticker))
            else:
                t_yf = yf.Ticker(ticker)
                exps = list(t_yf.options or [])
            if not exps:
                st.error("No expiries found.")
            else:
                e = st.selectbox("Choose expiry", exps)
                if e:
                    if implied_vol_from_chain is None:
                        st.error("implied_vol_from_chain not available in the package.")
                    else:
                        try:
                            df_iv, spot, T = implied_vol_from_chain(ticker, e, spot=None, r=r_pct/100.0)
                        except Exception:
                            # some implementations return df only; attempt to call defensively
                            try:
                                df_iv = implied_vol_from_chain(ticker, e, spot=None, r=r_pct/100.0)
                                spot = float(get_current_price(ticker) or 0.0)
                                import datetime
                                expiry_date = datetime.datetime.strptime(e, "%Y-%m-%d").date()
                                T = max((expiry_date - datetime.date.today()).days / 365.0, 1/365.0)
                            except Exception as ex:
                                st.error(f"Failed to fetch IV chain: {ex}")
                                df_iv = None
                                spot = None
                                T = None
                        if df_iv is None or df_iv.empty:
                            st.warning("No IV data for selected expiry.")
                        else:
                            st.caption(f"Spot={spot:.2f}, T≈{T:.4f} years")
                            st.dataframe(df_iv.head(50))

                            # IV curve
                            col = 'call_iv' if use_calls else 'put_iv'
                            if col in df_iv.columns:
                                fig = px.line(df_iv, x='strike', y=col, title=f"{ticker} {col} — {e}")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning(f"{col} column not present in IV table.")

                            st.download_button("Download IV table (CSV)", df_iv.to_csv(index=False).encode('utf-8'),
                                               file_name=f"{ticker}_{e}_iv.csv", mime="text/csv")

                            # Heston single-expiry calibration (if available)
                            if HestonParams is not None and calibrate_heston_to_iv is not None:
                                st.subheader("Heston Calibration (single expiry)")
                                mask = np.isfinite(df_iv['call_iv'].values if 'call_iv' in df_iv.columns else df_iv.iloc[:,0].values)
                                strikes = df_iv['strike'].values[mask] if 'strike' in df_iv.columns else df_iv.index.values[mask]
                                mkt_ivs = df_iv['call_iv'].values[mask] if 'call_iv' in df_iv.columns else df_iv.iloc[:,0].values[mask]

                                if len(strikes) >= 4:
                                    init = HestonParams(1.5, 0.04, 0.5, -0.6, 0.04)
                                    try:
                                        fitted, res = calibrate_heston_to_iv(
                                            spot, r_pct/100.0, T, strikes, mkt_ivs, init=init
                                        )
                                        st.success("Calibration complete.")
                                        st.write(f"**Fitted params**: κ={fitted.kappa:.4f}, θ={fitted.theta:.4f}, "
                                                 f"σ_v={fitted.sigma_v:.4f}, ρ={fitted.rho:.4f}, v0={fitted.v0:.4f}")
                                        model_ivs = heston_implied_vol_curve(spot, r_pct/100.0, T, strikes, fitted)
                                        table, rmse, mae = summarize_fit(strikes, mkt_ivs, model_ivs)
                                        st.caption(f"RMSE={rmse:.4g}, MAE={mae:.4g}")
                                        st.dataframe(table)

                                        fig2 = go.Figure()
                                        fig2.add_trace(go.Scatter(x=strikes, y=mkt_ivs, mode='markers+lines', name='Market IV'))
                                        fig2.add_trace(go.Scatter(x=strikes, y=model_ivs, mode='markers+lines', name='Heston IV'))
                                        fig2.update_layout(title=f"{ticker} — IV fit on {e}", xaxis_title="Strike", yaxis_title="IV")
                                        st.plotly_chart(fig2, use_container_width=True)
                                    except Exception as ex:
                                        st.error(f"Calibration failed: {ex}")
                                else:
                                    st.warning("Not enough valid IV points to calibrate (need ≥ 4).")
                            else:
                                st.info("Heston calibration utilities are not available (module import failed).")

        except Exception as ex:
            st.error(f"Error fetching expiries: {ex}")

    st.markdown("---")
    st.subheader("3D Surface (multiple expiries)")
    num_exp = st.slider("Number of expiries to include", 1, 8, 3)
    if st.button("Build 3D Surface"):
        try:
            if build_surface is None:
                st.error("build_surface not available in package.")
            else:
                expiries = []
                if fetch_expiries is not None:
                    expiries = list(fetch_expiries(ticker))[:num_exp]
                elif vol_fetch_expiries is not None:
                    expiries = list(vol_fetch_expiries(ticker))[:num_exp]
                else:
                    t = yf.Ticker(ticker)
                    expiries = list(t.options or [])[:num_exp]

                surf = build_surface(ticker, expiries, r=r_pct/100.0, use_calls=use_calls)
                if surf is None or (isinstance(surf, pd.DataFrame) and surf.empty):
                    st.warning("Surface is empty (insufficient market data).")
                else:
                    try:
                        surf = _normalize_surface_df(surf, use_calls=use_calls)
                        missing = [c for c in ("strike","T","iv") if c not in surf.columns]
                        if missing:
                            st.error(f"Surface is missing required columns: {missing}. Got columns: {list(surf.columns)}")
                        else:
                            st.dataframe(surf.head(20))
                            fig = px.scatter_3d(
                                surf, x='strike', y='T', z='iv', color='iv',
                                title=f"{ticker} Implied Vol Surface ({'Calls' if use_calls else 'Puts'})"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.download_button(
                                "Download surface CSV",
                                surf.to_csv(index=False).encode('utf-8'),
                                file_name=f"{ticker}_iv_surface.csv",
                                mime="text/csv"
                            )
                    except Exception as ex:
                        st.error(f"Failed building surface: {ex}")


                # surf = build_surface(ticker, expiries, r=r_pct/100.0, use_calls=use_calls)
                # if surf is None or surf.empty:
                #     st.warning("Surface is empty (insufficient market data).")
                # else:
                #     st.dataframe(surf.head(20))
                #     fig = px.scatter_3d(surf, x='strike', y='T', z='iv', color='iv',
                #                         title=f"{ticker} Implied Vol Surface ({'Calls' if use_calls else 'Puts'})")
                #     st.plotly_chart(fig, use_container_width=True)
                #     st.download_button("Download surface CSV", surf.to_csv(index=False).encode('utf-8'),
                #                        file_name=f"{ticker}_iv_surface.csv", mime="text/csv")
        except Exception as ex:
            st.error(f"Failed building surface: {ex}")

    st.markdown("---")
    st.subheader("Price with Heston (semi-closed form)")
    S0_hint = get_current_price(ticker)
    scol = st.columns(4)
    S0 = scol[0].number_input("Spot S0", min_value=0.01, value=float(S0_hint) if S0_hint else 100.0, step=0.01)
    K = scol[1].number_input("Strike", min_value=0.01, value=float(round(S0, 2)), step=0.01)
    T_years = scol[2].number_input("T (years)", min_value=1/365.0, value=0.5, step=0.01)
    r_use = scol[3].number_input("Risk-free r (%)", 0.0, 100.0, r_pct, step=0.1) / 100.0
    pcol = st.columns(5)
    kappa = pcol[0].number_input("κ", 0.01, 10.0, 1.5, step=0.01)
    theta = pcol[1].number_input("θ", 1e-4, 2.0, 0.04, step=0.001)
    sigma_v = pcol[2].number_input("σ_v", 1e-4, 3.0, 0.5, step=0.01)
    rho     = pcol[3].number_input("ρ", -0.999, 0.999, -0.6, step=0.001)
    v0      = pcol[4].number_input("v0", 1e-5, 2.0, 0.04, step=0.001)

    # if st.button("Price with Heston"):
    #     if HestonModel is None:
    #         st.error("HestonModel not available.")
    #     else:
    #         try:
    #             model = HestonModel(S0, K, T_years, r_use, kappa, theta, sigma_v, rho, v0)
    #             call = model.call_price()
    #             put = model.put_price()
    #             mm = st.columns(2)
    #             mm[0].metric("Call (Heston)", f"{call:.4f}")
    #             mm[1].metric("Put (Heston)", f"{put:.4f}")
    #         except Exception as ex:
    #             st.error(f"Heston pricing failed: {ex}")

    if st.button("Price with Heston"):
        if HestonModel is None:
            st.error("HestonModel not available.")
        else:
            try:
                # First try the package model
                model = HestonModel(S0, K, T_years, r_use, kappa, theta, sigma_v, rho, v0)
                call = float(model.call_price())
                put  = float(model.put_price())
            except Exception:
                call = np.nan
                put  = np.nan

            # Fallback if the model gave NaN/Inf
            if not np.isfinite(call):
                call = _heston_call_fallback(S0, K, T_years, r_use, kappa, theta, sigma_v, rho, v0)
            if not np.isfinite(put):
                put = _heston_put_from_call(call, S0, K, T_years, r_use)

            mm = st.columns(2)
            mm[0].metric("Call (Heston)", f"{call:.6f}")
            mm[1].metric("Put (Heston)", f"{put:.6f}")


# =========================
# Volatility Lab (Phase 7) — Vol-arb + Grid Search
# =========================
elif page == "Volatility Lab (Vol-Arb & Grid Search)":
    st.header("Volatility Lab — RV vs IV, Term Structure, Vol-Arb Backtest & Grid Search")

    # Inputs: ticker, expiry
    c1, c2 = st.columns([1,2])
    with c1:
        ticker_lab = st.text_input("Ticker", value="AAPL", key="vl_ticker")
    with c2:
        expiries = []
        try:
            if fetch_expiries is not None:
                expiries = list(fetch_expiries(ticker_lab))
            elif vol_fetch_expiries is not None:
                expiries = list(vol_fetch_expiries(ticker_lab))
            else:
                expiries = list(yf.Ticker(ticker_lab).options or [])
        except Exception:
            expiries = []
        expiry_select = st.selectbox("Expiry (choose)", expiries if expiries else ["(no expiries)"])

    st.markdown("### ATM term structure (market)")
    try:
        if aggregate_implied_term_structure is None:
            st.info("aggregate_implied_term_structure not available in package.")
        else:
            df_term = aggregate_implied_term_structure(ticker_lab, expiries=[expiry_select] if expiry_select and expiry_select != "(no expiries)" else None, r=0.01)
            if df_term is None or df_term.empty:
                st.warning("Term structure data not available for this ticker/expiry.")
            else:
                st.dataframe(df_term)
                st.line_chart(df_term.set_index('days_to_expiry')['atm_iv'])
    except Exception as ex:
        st.error(f"Term structure error: {ex}")

    st.markdown("---")
    st.subheader("Realized vs Implied (single expiry & ATM)")
    rv_col1, rv_col2 = st.columns(2)
    use_window = rv_col1.number_input("RV rolling window (days)", min_value=2, value=21)
    if st.button("Compute RV vs IV for selected expiry"):
        try:
            t = yf.Ticker(ticker_lab)
            hist = t.history(period="180d")
            if hist is None or hist.empty:
                st.error("No historical data to compute realized vol.")
            else:
                close = hist['Close'].dropna()
                # ATM IV for chosen expiry
                if implied_vol_from_chain is None:
                    st.error("implied_vol_from_chain not available.")
                else:
                    result = implied_vol_from_chain(ticker_lab, expiry_select, spot=None, r=0.01)
                    if result is None:
                        st.error("No IV data for expiry.")
                    else:
                        df_iv, underlying_price, rfr = result
                        if df_iv.empty:
                            st.error("No IV data for expiry.")
                        else:
                            spot = float(t.history(period='1d')['Close'].iloc[-1])
                            strikes = df_iv.index.values
                            atm_idx = int(np.argmin(np.abs(strikes - spot)))
                            atm_strike = strikes[atm_idx]
                            row = df_iv.loc[atm_strike]
                            atm_iv = row['call_iv'] if not pd.isna(row.get('call_iv', np.nan)) else row.get('put_iv', np.nan)
                            df_rv_iv = compute_rolling_realized_vs_implied(close, float(atm_iv), window_days=int(use_window))
                            st.line_chart(df_rv_iv[['realized_vol','implied_vol']])
                            st.dataframe(df_rv_iv.tail(50))
        except Exception as ex:
            st.error(f"RV vs IV failed: {ex}")

    st.markdown("---")
    st.subheader("Vol-Arb Backtest (Short ATM Straddle & Delta Hedge)")
    back_col = st.columns(3)
    n_paths = int(back_col[0].number_input("Sim paths", 100, 200000, 5000, step=100))
    n_steps = int(back_col[1].number_input("Steps", 2, 500, 50, step=1))
    threshold = float(back_col[2].number_input("IV/RV threshold to trade", 1.05, 3.0, 1.2, step=0.01))
    rebalance_every = int(st.number_input("Rebalance every (steps)", min_value=1, max_value=100, value=1))
    tc = float(st.number_input("Transaction cost per share ($)", 0.0, 5.0, 0.0, step=0.001))
    impact = float(st.number_input("Linear slippage per share ($)", 0.0, 1.0, 0.0, step=0.0001))
    quad = float(st.number_input("Quadratic impact coeff", 0.0, 10.0, 0.0, step=0.01))
    liq = float(st.number_input("Liquidity scale ($ notional)", min_value=1e3, max_value=1e9, value=1e6, step=1000.0))

    if st.button("Run vol-arb backtest"):
        if vol_arbitrage_strategy_backtest is None:
            st.error("vol_arbitrage_strategy_backtest is not available (module import failed).")
        else:
            try:
                res = vol_arbitrage_strategy_backtest(
                    ticker_lab, expiry_select, n_paths=n_paths, n_steps=n_steps,
                    threshold=threshold, rebalance_every=rebalance_every,
                    tc_per_share=tc, impact_per_share=impact, impact_coeff_quadratic=quad, liquidity_scale=liq, r=0.0
                )
                if not res.get('trade_signal', False):
                    st.warning(res.get('message', 'No trade signal'))
                else:
                    st.write("Strategy metadata:")
                    st.json({k: v for k, v in res.items() if k not in ('pnl_straddle','summary')})
                    st.write("Summary stats:")
                    st.dataframe(pd.DataFrame([res['summary']]).T.rename(columns={0:'Value'}))
                    fig, ax = plt.subplots(1,1,figsize=(8,4))
                    ax.hist(res['pnl_straddle'], bins=100, alpha=0.7)
                    ax.set_title('Delta-hedged short straddle pnl (discounted)')
                    st.pyplot(fig)
            except Exception as ex:
                st.error(f"Vol-arb failed: {ex}")

    st.markdown("---")
    st.subheader("Grid Search: thresholds × rebalance frequencies")
    grid_col1, grid_col2 = st.columns(2)
    thr_text = grid_col1.text_input("Thresholds (comma separated, e.g. 1.05,1.1,1.2)", value="1.05,1.1,1.2")
    reb_text = grid_col2.text_input("Rebalance frequencies (comma separated integers, e.g. 1,5,10)", value="1,5,10")
    gs_paths = st.number_input("Grid sim paths per cell (small for grid)", min_value=200, max_value=20000, value=2000, step=100)
    gs_steps = st.number_input("Grid steps per path", min_value=2, max_value=500, value=50, step=1)
    max_cells = st.number_input("Max grid cells", min_value=1, max_value=1000, value=40, step=1)

    if st.button("Run grid search"):
        if grid_search_vol_arb is None:
            st.error("grid_search_vol_arb not available (module import failed).")
        else:
            try:
                thresholds = [float(s.strip()) for s in thr_text.split(",") if s.strip()!='']
                rebalance_list = [int(s.strip()) for s in reb_text.split(",") if s.strip()!='']
                if len(thresholds) == 0 or len(rebalance_list) == 0:
                    st.error("Provide at least one threshold and one rebalance frequency.")
                else:
                    with st.spinner("Running grid search (may take a while)..."):
                        gs = grid_search_vol_arb(
                            ticker=ticker_lab, expiry=expiry_select,
                            thresholds=thresholds, rebalance_list=rebalance_list,
                            n_paths=int(gs_paths), n_steps=int(gs_steps),
                            tc_per_share=tc, impact_per_share=impact,
                            impact_coeff_quadratic=quad, liquidity_scale=liq,
                            r=0.0, max_cells=int(max_cells)
                        )
                    mean_mat = gs['mean_matrix']
                    std_mat = gs['std_matrix']
                    sharpe_mat = gs['sharpe_matrix']

                    def plot_heatmap(mat, title):
                        fig, ax = plt.subplots(1,1,figsize=(9,5))
                        im = ax.imshow(np.array(mat, dtype=float), origin='lower', aspect='auto', cmap='viridis')
                        ax.set_xticks(range(len(thresholds)))
                        ax.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
                        ax.set_yticks(range(len(rebalance_list)))
                        ax.set_yticklabels([str(r) for r in rebalance_list])
                        ax.set_xlabel("Threshold")
                        ax.set_ylabel("Rebalance")
                        ax.set_title(title)
                        fig.colorbar(im, ax=ax)
                        st.pyplot(fig)

                    st.subheader("Mean discounted P&L heatmap")
                    plot_heatmap(mean_mat, "Mean discounted P&L")
                    st.subheader("Std discounted P&L heatmap")
                    plot_heatmap(std_mat, "Std discounted P&L")
                    st.subheader("Sharpe-like (mean/std) heatmap")
                    plot_heatmap(sharpe_mat, "Mean/Std (Sharpe-like)")
                    st.success("Grid search finished.")
            except Exception as ex:
                st.error(f"Grid search failed: {ex}")

# =========================
# Advanced: FFT / LocalVol / Hedging (Phase 8)
# =========================
elif page == "Advanced: FFT / LocalVol / Hedging":
    st.header("Advanced Pricing & Vol Dynamics — Carr–Madan FFT, Dupire LocalVol, MV Hedging")

    st.subheader("Carr–Madan FFT demo (Black–Scholes test)")
    c1, c2, c3 = st.columns(3)
    S0 = float(c1.number_input("Spot S0", value=100.0))
    r_fft = float(c2.number_input("Risk-free rate (%)", value=1.0)) / 100.0
    sigma_fft = float(c3.number_input("BS vol (%)", value=20.0)) / 100.0
    T_days = st.number_input("Days to maturity", min_value=1, value=90)
    T_fft = float(T_days) / 365.0
    alpha = st.number_input("Damping alpha", value=1.5, step=0.1)
    eta = st.number_input("FFT eta (freq spacing)", value=0.25, step=0.01)
    N_fft = int(st.number_input("FFT N (power of two)", value=2**10))
    B = float(st.number_input("FFT B (log-strike bound)", value=80.0))

    if st.button("Run Carr–Madan FFT (BS)"):
        if carr_madan_fft is None or cf_bs is None:
            st.error("Carr–Madan or CF not available in package.")
        else:
            try:
                cf = lambda u: cf_bs(u, S0, r_fft, sigma_fft, T_fft)
                strikes, calls = carr_madan_fft(cf, S0=S0, r=r_fft, T=T_fft, alpha=alpha, eta=eta, N=N_fft, B=B)
                df_fft = pd.DataFrame({'strike': strikes, 'call_price_fft': calls})
                # compute BS prices for comparison
                bs_prices = np.array([BlackScholesModel(S0, K, int(T_days), r_fft, sigma_fft).calculate_option_price('Call Option') for K in strikes])
                df_fft['call_price_bs'] = bs_prices
                st.write("FFT vs Black-Scholes (sample):")
                st.dataframe(df_fft.head(30))
                fig = px.line(df_fft, x='strike', y=['call_price_fft','call_price_bs'], labels={'value':'Price','variable':'Method'})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as ex:
                st.error(f"FFT pricing failed: {ex}")

    st.markdown("---")
    st.subheader("Dupire Local Vol (demo from synthetic IV grid)")
    t_loc1, t_loc2 = st.columns(2)
    loc_ticker = t_loc1.text_input("Ticker (used only for spot hint)", value="AAPL", key="local_ticker")
    grid_build = t_loc2.button("Build demo synthetic IV grid & compute local vol")
    if grid_build:
        if dupire_local_vol is None:
            st.error("Dupire local vol module not available.")
        else:
            try:
                spot_hint = get_current_price(loc_ticker) or 100.0
                strikes = np.linspace(0.6*spot_hint, 1.4*spot_hint, 25)
                maturities_days = [30, 60, 90, 180]
                maturities = np.array(maturities_days) / 365.0
                # synthetic skewed IV surface
                iv_grid = np.zeros((len(maturities), len(strikes)))
                for i,Tt in enumerate(maturities):
                    for j,K in enumerate(strikes):
                        iv_grid[i,j] = 0.12 + 0.18 * abs(K - spot_hint) / spot_hint + 0.05 * (0.5 - (i/len(maturities)))
                local_vol, price_surface = dupire_local_vol(strikes, maturities, iv_grid, spot_hint, r=r_fft)
                fig, ax = plt.subplots(1,1,figsize=(10,4))
                im = ax.imshow(local_vol, origin='lower', aspect='auto', cmap='inferno')
                ax.set_title("Local vol surface (rows=maturities, cols=strikes)")
                fig.colorbar(im, ax=ax)
                st.pyplot(fig)
            except Exception as ex:
                st.error(f"Dupire local vol failed: {ex}")

    st.markdown("---")
    st.subheader("Mean–Variance Hedge (MC estimate)")
    h1, h2, h3 = st.columns(3)
    S0_h = float(h1.number_input("Spot S0", value=100.0, key="mv_S0"))
    K_h = float(h2.number_input("Strike", value=100.0, key="mv_K"))
    sigma_true_h = float(h3.number_input("True sigma (%)", value=20.0, key="mv_sigma"))/100.0
    T_h = int(st.number_input("Days to maturity", min_value=1, value=30, key="mv_days"))/365.0
    if st.button("Estimate MV hedge (MC)"):
        if compute_mv_hedge_ratio_mc is None:
            st.error("MV hedge utility not available.")
        else:
            try:
                h_ratio, stats = compute_mv_hedge_ratio_mc(S0_h, K_h, r_fft, sigma_true_h, T_h, option_type='call', n_paths=5000, n_steps=50, seed=42)
                st.write("MV hedge ratio (units underlying per short option):", round(h_ratio,6))
                st.write("Cov/Var diagnostic:", stats)
            except Exception as ex:
                st.error(f"MV hedge estimation failed: {ex}")

    st.markdown("---")
    st.subheader("Realized vs Implied surface (selected expiries)")
    rv_ticker = st.text_input("Ticker for RV vs IV", value="AAPL", key="adv_rvticker")
    exp_input = st.text_input("Expiries list (comma separated yyyy-mm-dd)", value="", key="adv_expiries")
    if st.button("Compute RV vs IV for expiries") and exp_input.strip() != "":
        try:
            exps = [s.strip() for s in exp_input.split(",") if s.strip()!='']
            if compute_realized_vs_implied_surface is None:
                st.error("compute_realized_vs_implied_surface not available.")
            else:
                df_cmp = compute_realized_vs_implied_surface(rv_ticker, exps, window_days=21, r=r_fft)
                if df_cmp.empty:
                    st.warning("No data available for the given ticker/expiries.")
                else:
                    st.dataframe(df_cmp)
                    st.line_chart(df_cmp.set_index('days_to_expiry')[['atm_iv','realized_vol']])
        except Exception as ex:
            st.error(f"RV vs IV failed: {ex}")
# ======= Paper Trading (Runner) page snippet =======
elif page == "Paper Trading (Runner)":
    st.header("📈 Paper Trading & Runner (DB logging)")
    st.write("Start a paper-run with a strategy plugin and persist trades to Postgres.")

    st.markdown("""
    **About Short Straddle Delta Hedge Strategy**  
    The Short Straddle involves selling both a call and a put option at the same strike price (usually ATM).  
    It profits from low volatility (collecting premiums) but is exposed to unlimited risk if the underlying moves significantly.  
    To reduce directional risk, we apply **delta hedging**, dynamically adjusting the underlying stock position so the overall delta of the portfolio stays close to zero.
    """)

    tickers_input = st.text_input("Tickers (comma separated)", value="AAPL")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    tc = st.number_input("Transaction cost per share ($)", 0.0, 10.0, 0.001, step=0.001)
    impact = st.number_input("Linear impact per share ($)", 0.0, 1.0, 0.0)
    quad = st.number_input("Quadratic impact coeff", 0.0, 10.0, 0.0)
    portfolio_name = st.text_input("Portfolio name", value="demo_portfolio")

    if 'runner_persist' not in st.session_state:
        from option_pricing.adapters import MarketAdapter
        from option_pricing.execution import SimpleExecutionEngine
        from option_pricing.runner_db_glue import RunnerWithPersistence
        ma = MarketAdapter()  # uses env keys if present
        exec_engine = SimpleExecutionEngine(ma, tc_per_share=tc, impact_per_share=impact, impact_coeff_quadratic=quad)
        st.session_state.runner_persist = RunnerWithPersistence(ma, exec_engine, portfolio_name=portfolio_name)

    st.markdown("### Strategy Management")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Register example ShortStraddle strategy"):
            try:
                from strategies.short_straddle_delta_hedge import ShortStraddleDeltaHedge
                strat = ShortStraddleDeltaHedge({"ticker": tickers[0], "notional": 10000.0})
                st.session_state.runner_persist.register("short_straddle", strat)
                st.success("Registered short_straddle")
            except Exception as e:
                st.error(f"Register failed: {e}")

    with col2:
        if st.button("Start runner"):
            try:
                st.session_state.runner_persist.start_all(tickers, interval_sec=5, rebalance_freq_sec=30)
                st.success("Runner started")
            except Exception as e:
                st.error(f"Start failed: {e}")

    with col3:
        if st.button("Stop runner"):
            try:
                st.session_state.runner_persist.stop_all()
                st.success("Runner stopped")
            except Exception as e:
                st.error(f"Stop failed: {e}")

    st.markdown("---")
    st.subheader("Recent Trades (Persisted to DB)")
    try:
        df_trades = st.session_state.runner_persist.trades_df(limit=200)
        st.dataframe(df_trades, use_container_width=True)
        if not df_trades.empty:
            st.download_button("Download trades CSV", df_trades.to_csv(index=False).encode('utf-8'), file_name="trades.csv")
    except Exception as e:
        st.info("No trades yet or DB not configured.")


# ======= Risk Dashboard =======
elif page == "Risk Dashboard":
    st.header("🛡️ Risk Dashboard")
    st.markdown("""
    This page allows you to analyze **portfolio risk exposure** by applying spot shocks to your positions 
    and computing NAV under different market scenarios.

    **Expected Input (CSV format):**
    - Columns:  
      - `type` (call/put)  
      - `K` (strike)  
      - `T` (time-to-maturity in years)  
      - `r` (risk-free rate)  
      - `sigma` (volatility)  
      - `spot` (underlying spot price)  
    - Each row represents one option position.
    """)

    uploaded = st.file_uploader("Upload positions CSV", type="csv")
    if uploaded:
        df_pos = pd.read_csv(uploaded)
        st.write("### Uploaded Positions")
        st.dataframe(df_pos, use_container_width=True)

        shocks = np.linspace(-0.2, 0.2, 41)
        from option_pricing.risk import sweep_spot_scenario
        from option_pricing.BlackScholesModel import BlackScholesModel

        def pricing_fn(pos, spot):
            p = BlackScholesModel(
                spot,
                float(pos['K']),
                int(pos['T'] * 365),
                float(pos.get('r', 0.0)),
                float(pos.get('sigma', 0.2))
            )
            return p.calculate_option_price("Call Option" if pos.get('type', 'call') == 'call' else 'Put Option')

        rows = df_pos.to_dict('records')
        df_sweep = sweep_spot_scenario(rows, spot_base=float(rows[0].get('spot', 100.0)), shocks=shocks, pricing_fn=pricing_fn)

        st.markdown("### NAV under Spot Shocks")
        fig = px.line(df_sweep, x='shock_pct', y='nav', title="Scenario NAV vs Spot Shock")
        st.plotly_chart(fig, use_container_width=True)


# ======= Performance =======
elif page == "Performance":
    st.header("📊 Performance Report")
    st.markdown("""
    This page provides **performance analytics** for your trading strategy.  
    You can upload NAV time-series and trade logs to generate key metrics.

    **Expected Inputs:**
    1. NAV CSV with columns:  
       - `time` (datetime)  
       - `nav` (portfolio NAV at each timestamp)  
    2. Trades CSV (optional) with typical columns such as:  
       - `time`, `ticker`, `quantity`, `price`, `side` (buy/sell)
    """)

    uploaded_nav = st.file_uploader("Upload NAV time series (csv with 'time','nav')", type="csv")
    uploaded_trades = st.file_uploader("Upload trades csv", type='csv')

    if uploaded_nav:
        nav = pd.read_csv(uploaded_nav, parse_dates=['time']).set_index('time')['nav']
        from option_pricing.performance import performance_report
        trades = pd.read_csv(uploaded_trades) if uploaded_trades else pd.DataFrame()
        report = performance_report(nav, trades)

        st.subheader("Performance Metrics")
        st.json(report)

        st.subheader("NAV Over Time")
        st.line_chart(nav)


elif page == "About & Notes":
    st.header("About the Developer")
    st.markdown(
        """
        ### 👋 Hi, I’m **Gaurav Poddar**
        - 📍 Based in New York City  
        - 🎓 Studying Computer Science at **New York University (Tandon)** with additional coursework at **Stern School of Business**  
        - ✉️ Email: [gp2610@nyu.edu](mailto:gp2610@nyu.edu) | [gp2610@stern.nyu.edu](mailto:gp2610@stern.nyu.edu)  
        - 🔗 [LinkedIn](https://www.linkedin.com/in/gauravpoddar-gp13/) | [GitHub](https://github.com/Gaurav06Poddar)  

        ---
        ### 📖 My Story
        I’ve always been fascinated by the intersection of **mathematics, programming, and markets**.  
        What started as a curiosity — *“How do quants actually price options?”* — grew into a passion project where I built this **quant-research and trading lab** from scratch.

        This project isn’t just code — it’s my way of showing that I can:
        - **Think like a quant researcher** (deriving, testing, and comparing models),  
        - **Code like a quant developer** (building scalable, tested, production-style infrastructure),  
        - And **analyze like a trader** (asking: *does this model actually give an edge?*).  

        I designed this app as a live, interactive **portfolio piece** for recruiters — if you’re reading this, you’re literally seeing how I think, build, and ship.

        ---
        ### 🛠️ What This App Can Do

        #### 📌 Core Models
        - **Black–Scholes**: closed-form pricing + Greeks (Δ, Γ, Θ, ρ, Vega).  
        - **Binomial Tree**: European & American, with early-exercise logic.  
        - **Monte Carlo (European)**: variance reduction (antithetic / control) + pathwise/LR Greeks with standard errors.  
        - **Longstaff–Schwartz (American MC)**: regression-based continuation values.  
        - **Asian Options (Arithmetic)**: Monte Carlo with control variates.  

        #### 📌 Volatility Tools
        - Synthetic implied volatility surface builder.  
        - Real-market implied volatility extraction (via **Yahoo Finance options chains**).  
        - Volatility Lab: realized vs implied, delta-hedged vol-arb backtests, grid search over thresholds/rebalances, execution cost modeling.  

        #### 📌 Advanced & Cutting Edge
        - **Heston Model**: calibration (single expiry) + semi-closed form pricing.  
        - **Carr–Madan FFT pricer**: demo vs Black–Scholes characteristic function.  
        - **Dupire Local Vol Surface**: build from implied IV grids.  
        - **Mean–Variance Hedging**: practical helper for risk-optimal hedges.  

        ---
        ### 💡 Notes & Tips
        - Heavy Monte Carlo runs can be slow in Streamlit — small numbers for demo, larger for offline runs.  
        - Local vol estimation is numerically sensitive — smoothing implied vols helps.  
        - Yahoo Finance data isn’t perfect — bid/ask gaps and missing strikes may appear.  
        - Recruiter Demo Tip:  
            Compare BS vs FFT, show Monte Carlo Greeks with SE, and demo vol-arb grid heatmaps — these highlight both **quant insight** and **engineering skill**.

        ---
        ### 📐 Modules & Formulae

        - **Black–Scholes Model**
          - European Call: C = S₀·Φ(d₁) − K·e^(−rT)·Φ(d₂)  
          - European Put: P = K·e^(−rT)·Φ(−d₂) − S₀·Φ(−d₁)  
          - where d₁ = [ln(S₀/K) + (r + σ²/2)·T] / (σ√T),  d₂ = d₁ − σ√T  
          - Greeks:  
            - Delta (Δ) = Φ(d₁) (Call), Φ(d₁)−1 (Put)  
            - Gamma (Γ) = φ(d₁) / (S₀σ√T)  
            - Vega (ν) = S₀·φ(d₁)√T  
            - Theta (Θ) = −(S₀φ(d₁)σ)/(2√T) − rK·e^(−rT)Φ(d₂) (Call), etc.  
            - Rho (ρ) = K·T·e^(−rT)Φ(d₂) (Call), −K·T·e^(−rT)Φ(−d₂) (Put)

        - **Binomial Tree Model**
          - Stock price evolution: Sᵢⱼ = S₀·uʲ·d⁽ⁱ⁻ʲ⁾  
          - u = e^(σ√Δt),  d = 1/u  
          - Risk-neutral probability: p = (e^(rΔt) − d) / (u − d)  
          - Option value: V = e^(−rΔt)[p·Vᵤ + (1−p)·V_d]  
          - Supports early-exercise for American options.

        - **Monte Carlo Simulation**
          - Path generation: S_T = S₀·exp((r−½σ²)T + σ√T·Z), with Z~N(0,1)  
          - Payoff: C = e^(−rT)·max(S_T − K, 0)  
          - Variance reduction: Antithetic variates (use −Z), Control variates (use BS closed form), Greeks via Pathwise & Likelihood Ratio.  

        - **Longstaff–Schwartz (American Monte Carlo)**
          - Regression of continuation value vs basis functions of underlying.  
          - Exercise decision = max(immediate payoff, continuation).  

        - **Asian Options (Arithmetic Average, MC)**
          - Average: A = (1/m)∑S_t  
          - Payoff: Call = e^(−rT)·max(A − K, 0).  
          - Control variate: Use geometric Asian (with closed form) to reduce variance.  

        - **Implied Volatility (IV)**
          - Find σ such that BS_price(S₀, K, T, σ) = Market Price.  
          - Root-finding via Brent’s method / Newton–Raphson.  
          - Real-market IV surface extracted from Yahoo Finance option chains.  

        - **Heston Model**
          - Dynamics:  
            dS_t = μS_t dt + √v_t S_t dW₁t  
            dv_t = κ(θ − v_t)dt + σ√v_t dW₂t, corr(dW₁,dW₂) = ρ  
          - Semi-closed form pricing via characteristic function:  
            C = S₀·P₁ − K·e^(−rT)·P₂,  
            where P₁, P₂ = (1/2) + (1/π)∫Re(e^(−iu lnK)·f_j(u)/(iu)) du.  

        - **Carr–Madan FFT**
          - Option price recovered from characteristic function φ(u):  
            C(K) = e^(−αk)/π ∫ Re(e^(−iuk) φ(u−(i(α+1)))/(α²+α−u²+i(2α+1)u)) du.  

        - **Dupire Local Volatility**
          - Local variance: σ²(K,T) = (∂C/∂T + rK∂C/∂K) / (½K²∂²C/∂K²).  
          - Requires smoothing IV surface before differentiation.  

        - **Risk & Performance Dashboards**
          - PnL attribution: ΔPnL ≈ Δ·ΔS + ½Γ·(ΔS)² + Vega·Δσ.  
          - Portfolio VaR (Variance–Covariance): VaR = z·√(wᵀΣw).  
          - Sharpe Ratio = (E[R_p − R_f]) / σ_p.  
          - Sortino Ratio = (E[R_p − R_f]) / σ_downside.  
          - Max Drawdown = max(peak − trough)/peak.  

        - **Paper Trading Module**
          - Executes simulated trades using market data (Yahoo Finance).  
          - Tracks positions, cash balance, realized/unrealized PnL.  
          - Allows testing of strategies (delta-hedging, vol-arb) with realistic slippage models.

        ---
        ### 🚀 What’s Next
        If you’re a recruiter:  
        Imagine extending this into a **full live backtester** with execution hooks — the exact kind of infrastructure a quant desk would use.

        That’s where I want to take this next.  
        And maybe… with your team, I can.
        """
    )

