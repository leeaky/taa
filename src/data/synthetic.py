# =============================================================================
# SYNTHETIC DATA GENERATOR
# Produces realistic macro and market data for testing and showcase.
# Mirrors the exact schema that fetcher.py produces so all downstream
# modules work identically whether data is live or synthetic.
#
# Uses regime-aware simulation: cycles through Goldilocks → Reflation →
# Stagflation → Deflation with realistic transitions and noise.
# NOT for production — clearly labelled in dashboard when active.
# =============================================================================

import numpy as np
import pandas as pd
from datetime import datetime

SEED = 42


def _ar1(n: int, phi: float, sigma: float, mu: float = 0.0, rng=None) -> np.ndarray:
    """AR(1) process: x_t = mu + phi*(x_{t-1}-mu) + sigma*e_t"""
    if rng is None:
        rng = np.random.default_rng(SEED)
    x = np.zeros(n)
    x[0] = mu
    e = rng.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + e[t]
    return x


def generate_regime_cycle(n_months: int, rng=None) -> np.ndarray:
    """
    Generate a sequence of regime labels cycling through the four quadrants.
    Each regime lasts 12-24 months with some randomness.
    Returns array of strings.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    regimes = ["Goldilocks", "Reflation", "Stagflation", "Deflation"]
    result = []
    i = 0
    regime_idx = 0
    while i < n_months:
        duration = int(rng.uniform(12, 28))
        label = regimes[regime_idx % 4]
        result.extend([label] * min(duration, n_months - i))
        i += duration
        regime_idx += 1
    return np.array(result[:n_months])


def _growth_signal(regime: str) -> float:
    return {"Goldilocks": 0.4, "Reflation": 0.3, "Stagflation": -0.3, "Deflation": -0.5}[regime]


def _inflation_signal(regime: str) -> float:
    return {"Goldilocks": -0.2, "Reflation": 0.5, "Stagflation": 0.4, "Deflation": -0.4}[regime]


def generate_synthetic_data(start: str = "1990-01-01") -> dict:
    """
    Generate synthetic data matching fetcher.py output schema.

    Returns
    -------
    dict : {series_id: {"data": pd.Series, "source": str,
                         "fetched_at": datetime, "frequency": str}}
    """
    rng = np.random.default_rng(SEED)
    fetched_at = datetime.utcnow()

    dates_monthly = pd.date_range(start=start, end=datetime.utcnow(), freq="ME")
    dates_daily   = pd.date_range(start=start, end=datetime.utcnow(), freq="B")
    dates_qtrly   = pd.date_range(start=start, end=datetime.utcnow(), freq="QE")

    n_m = len(dates_monthly)
    n_d = len(dates_daily)
    n_q = len(dates_qtrly)

    # Generate regime cycle for full history
    regime_cycle = generate_regime_cycle(n_m, rng)

    # Growth and inflation signals from regime
    g_sig = np.array([_growth_signal(r) for r in regime_cycle])
    i_sig = np.array([_inflation_signal(r) for r in regime_cycle])

    # --- FRED series ---------------------------------------------------------

    # GDP (quarterly) — real GDP growth in levels, trend ~15000 + growth
    gdp_growth = 0.006 + g_sig[::3][:n_q] * 0.003 + rng.normal(0, 0.005, n_q)
    gdp_levels = 15000 * np.cumprod(1 + gdp_growth)
    gdp = pd.Series(gdp_levels[:n_q], index=dates_qtrly[:n_q], name="GDP")

    # CPI (monthly) — YoY% implied from levels
    cpi_mom = 0.002 + i_sig * 0.002 + _ar1(n_m, 0.85, 0.001, rng=rng)
    cpi_levels = 150 * np.cumprod(1 + cpi_mom)
    cpi = pd.Series(cpi_levels, index=dates_monthly, name="CPIAUCSL")

    # PCE — similar to CPI with slight lag
    pce_levels = 130 * np.cumprod(1 + cpi_mom * 0.9 + rng.normal(0, 0.0005, n_m))
    pce = pd.Series(pce_levels, index=dates_monthly, name="PCEPI")

    # 10y-2y yield spread — growth leading indicator
    spread = g_sig * 1.5 + _ar1(n_m, 0.92, 0.15, rng=rng)
    t10y2y = pd.Series(spread, index=dates_monthly, name="T10Y2Y")

    # 5yr5yr forward inflation expectation
    fwd_inflation = 2.0 + i_sig * 0.8 + _ar1(n_m, 0.88, 0.1, rng=rng)
    t5yifr = pd.Series(fwd_inflation, index=dates_monthly, name="T5YIFR")

    # TIPS real yield
    tips = -0.5 + g_sig * 0.5 - i_sig * 0.3 + _ar1(n_m, 0.9, 0.1, rng=rng)
    dfii5 = pd.Series(tips, index=dates_monthly, name="DFII5")

    # M2 (monthly) — loose in Goldilocks/Reflation, tight in Stagflation/Deflation
    liq_sig = np.where(np.isin(regime_cycle, ["Goldilocks", "Reflation"]), 0.006, 0.002)
    m2_growth = liq_sig + rng.normal(0, 0.003, n_m)
    m2_levels = 8000 * np.cumprod(1 + m2_growth)
    m2 = pd.Series(m2_levels, index=dates_monthly, name="M2SL")

    # Fed balance sheet
    fed_growth = liq_sig * 1.5 + rng.normal(0, 0.005, n_m)
    fed_levels = 2000 * np.cumprod(1 + fed_growth)
    walcl = pd.Series(fed_levels, index=dates_monthly, name="WALCL")

    # USD index — strengthens in Deflation/Stagflation
    usd_sig = np.where(np.isin(regime_cycle, ["Deflation", "Stagflation"]), 0.3, -0.1)
    usd = 100 + _ar1(n_m, 0.95, 1.5, mu=0, rng=rng) + usd_sig * 3
    dtwexbgs = pd.Series(usd, index=dates_monthly, name="DTWEXBGS")

    # OECD CLI — leading growth indicator
    cli_base = 100 + g_sig * 3 + _ar1(n_m, 0.9, 0.5, rng=rng)
    oecd_cli = pd.Series(cli_base, index=dates_monthly, name="USALOLITONOSTSAM")

    # HY credit spread — tight in Goldilocks, wide in Stagflation/Deflation
    hy_sig = np.where(np.isin(regime_cycle, ["Goldilocks", "Reflation"]), -1.5, 2.5)
    hy_spread = 4.5 + hy_sig + _ar1(n_m, 0.88, 0.3, rng=rng)
    hy_spread = np.maximum(hy_spread, 1.5)
    baml = pd.Series(hy_spread, index=dates_monthly, name="BAMLH0A0HYM2")

    # --- Yahoo series (daily) ------------------------------------------------

    # Interpolate regime to daily
    regime_monthly_idx = {d: r for d, r in zip(dates_monthly, regime_cycle)}

    def regime_at_daily(d):
        # Find nearest monthly date
        nearest = min(dates_monthly, key=lambda m: abs((m - d).days))
        return regime_monthly_idx.get(nearest, "Goldilocks")

    daily_regimes = [regime_at_daily(d) for d in dates_daily]

    def _equity_returns(regime_series, base=100, annual_vol=0.16, rng=None):
        if rng is None:
            rng = np.random.default_rng(SEED)
        regime_drift = {
            "Goldilocks": 0.10 / 252,
            "Reflation":  0.06 / 252,
            "Stagflation": -0.05 / 252,
            "Deflation":  -0.10 / 252,
        }
        n = len(regime_series)
        daily_vol = annual_vol / np.sqrt(252)
        drifts = np.array([regime_drift[r] for r in regime_series])
        ret = drifts + rng.normal(0, daily_vol, n)
        prices = base * np.cumprod(1 + ret)
        return prices

    # VIX — high in bad regimes
    vix_base = {"Goldilocks": 14, "Reflation": 18, "Stagflation": 28, "Deflation": 35}
    vix_vals = np.array([vix_base[r] for r in daily_regimes], dtype=float)
    vix_vals += _ar1(n_d, 0.92, 2.0, rng=rng)
    vix_vals = np.maximum(vix_vals, 9)
    vix = pd.Series(vix_vals, index=dates_daily[:n_d], name="^VIX")

    # S&P 500
    spx = pd.Series(
        _equity_returns(daily_regimes, base=300, annual_vol=0.16, rng=rng),
        index=dates_daily[:n_d], name="^GSPC"
    )

    # Global equities (VT) — slightly more volatile
    vt = pd.Series(
        _equity_returns(daily_regimes, base=50, annual_vol=0.17, rng=rng),
        index=dates_daily[:n_d], name="VT"
    )

    # EM equities — more volatile, underperforms in USD strength
    em_drift_adj = np.where(
        np.array([r in ["Deflation", "Stagflation"] for r in daily_regimes]),
        -0.03 / 252, 0.02 / 252
    )
    em_ret = (np.array([{"Goldilocks": 0.12, "Reflation": 0.08,
                          "Stagflation": -0.08, "Deflation": -0.15}[r]
                         for r in daily_regimes]) / 252
              + em_drift_adj
              + rng.normal(0, 0.20 / np.sqrt(252), n_d))
    eem = pd.Series(40 * np.cumprod(1 + em_ret), index=dates_daily[:n_d], name="EEM")

    # Bonds (AGG) — rallies in Deflation
    bond_drift = {"Goldilocks": 0.03, "Reflation": 0.01, "Stagflation": -0.02, "Deflation": 0.08}
    bond_ret = (np.array([bond_drift[r] for r in daily_regimes]) / 252
                + rng.normal(0, 0.04 / np.sqrt(252), n_d))
    agg = pd.Series(80 * np.cumprod(1 + bond_ret), index=dates_daily[:n_d], name="AGG")

    # Long duration (TLT)
    tlt_drift = {"Goldilocks": 0.02, "Reflation": -0.02, "Stagflation": -0.05, "Deflation": 0.15}
    tlt_ret = (np.array([tlt_drift[r] for r in daily_regimes]) / 252
               + rng.normal(0, 0.10 / np.sqrt(252), n_d))
    tlt = pd.Series(90 * np.cumprod(1 + tlt_ret), index=dates_daily[:n_d], name="TLT")

    # HYG
    hyg_drift = {"Goldilocks": 0.07, "Reflation": 0.05, "Stagflation": -0.04, "Deflation": -0.08}
    hyg_ret = (np.array([hyg_drift[r] for r in daily_regimes]) / 252
               + rng.normal(0, 0.08 / np.sqrt(252), n_d))
    hyg = pd.Series(70 * np.cumprod(1 + hyg_ret), index=dates_daily[:n_d], name="HYG")

    # LQD
    lqd_drift = {"Goldilocks": 0.05, "Reflation": 0.03, "Stagflation": -0.02, "Deflation": 0.04}
    lqd_ret = (np.array([lqd_drift[r] for r in daily_regimes]) / 252
               + rng.normal(0, 0.06 / np.sqrt(252), n_d))
    lqd = pd.Series(100 * np.cumprod(1 + lqd_ret), index=dates_daily[:n_d], name="LQD")

    # Commodities (DJP) — strong in Reflation/Stagflation
    com_drift = {"Goldilocks": 0.04, "Reflation": 0.10, "Stagflation": 0.08, "Deflation": -0.06}
    com_ret = (np.array([com_drift[r] for r in daily_regimes]) / 252
               + rng.normal(0, 0.18 / np.sqrt(252), n_d))
    djp = pd.Series(20 * np.cumprod(1 + com_ret), index=dates_daily[:n_d], name="DJP")

    # Gold — safe haven, strong in Deflation and late Stagflation
    gld_drift = {"Goldilocks": 0.04, "Reflation": 0.06, "Stagflation": 0.08, "Deflation": 0.12}
    gld_ret = (np.array([gld_drift[r] for r in daily_regimes]) / 252
               + rng.normal(0, 0.12 / np.sqrt(252), n_d))
    gld = pd.Series(60 * np.cumprod(1 + gld_ret), index=dates_daily[:n_d], name="GLD")

    # --- Package into fetcher-compatible dict --------------------------------
    fred_series = {
        "GDP":             (gdp, "Q"),
        "CPIAUCSL":        (cpi, "M"),
        "PCEPI":           (pce, "M"),
        "T10Y2Y":          (t10y2y, "M"),
        "T5YIFR":          (t5yifr, "M"),
        "DFII5":           (dfii5, "M"),
        "M2SL":            (m2, "M"),
        "WALCL":           (walcl, "M"),
        "DTWEXBGS":        (dtwexbgs, "M"),
        "USALOLITONOSTSAM":(oecd_cli, "M"),
        "BAMLH0A0HYM2":    (baml, "M"),
    }

    yahoo_series = {
        "^VIX":  (vix, "D"),
        "^GSPC": (spx, "D"),
        "VT":    (vt, "D"),
        "EEM":   (eem, "D"),
        "AGG":   (agg, "D"),
        "TLT":   (tlt, "D"),
        "HYG":   (hyg, "D"),
        "LQD":   (lqd, "D"),
        "DJP":   (djp, "D"),
        "GLD":   (gld, "D"),
    }

    results = {}
    for sid, (s, freq) in fred_series.items():
        results[sid] = {
            "data":       s,
            "source":     "FRED_SYNTHETIC",
            "fetched_at": fetched_at,
            "frequency":  freq,
        }
    for sid, (s, freq) in yahoo_series.items():
        results[sid] = {
            "data":       s,
            "source":     "YAHOO_SYNTHETIC",
            "fetched_at": fetched_at,
            "frequency":  freq,
        }

    return results, regime_cycle, dates_monthly
