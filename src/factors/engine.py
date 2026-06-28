# =============================================================================
# MODULE 2: FACTOR ENGINE
# Transforms vintage-adjusted monthly data into four normalised factor scores
# across all configured z-score windows simultaneously.
#
# Factors:
#   Growth     — leading (60%) + coincident (40%) sub-components
#   Inflation  — CPI, PCE, breakevens, commodity momentum
#   Liquidity  — multi-central-bank aware (thin on public data, flagged)
#   Risk Appetite — VIX, put/call ratio, credit spreads, EM/DM relative, equity trend
#
# Each factor is computed at every configured window (36m, 60m, 120m, 240m, expanding).
# Output: wide dataframe with columns factor_Xm (e.g. growth_60m) for each window.
# Plus consensus score (std dev across windows) per factor at each date.
# =============================================================================

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Z-SCORE UTILITIES
# All z-scores are computed on data available up to each point in time only.
# Never full-sample. Rolling = fixed window. Expanding = growing window from start.
# =============================================================================

def zscore_rolling(s: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score: (x - rolling_mean) / rolling_std
    Uses only data available up to each point. Min periods = half window.
    """
    min_p = max(12, window // 2)
    mean = s.rolling(window=window, min_periods=min_p).mean()
    std  = s.rolling(window=window, min_periods=min_p).std()
    return ((s - mean) / std.replace(0, np.nan)).rename(s.name)


def zscore_expanding(s: pd.Series, min_periods: int = 36) -> pd.Series:
    """
    Expanding z-score: uses all history available up to each point.
    Default min_periods overridden by config.ZSCORE_MIN_PERIODS when called directly.
    """
    mean = s.expanding(min_periods=min_periods).mean()
    std  = s.expanding(min_periods=min_periods).std()
    return ((s - mean) / std.replace(0, np.nan)).rename(s.name)


def compute_all_windows(s: pd.Series, windows: list, expanding: bool) -> dict:
    """
    Compute z-scores for all windows. Returns {label: pd.Series}.
    Labels: '36m', '60m', '120m', '240m', 'expanding'
    """
    result = {}
    for w in windows:
        result[f"{w}m"] = zscore_rolling(s, w)
    if expanding:
        result["expanding"] = zscore_expanding(s)
    return result


def composite_zscore(series_dict: dict, weights: Optional[dict] = None) -> pd.Series:
    """
    Equal-weighted (or custom-weighted) composite of multiple z-scored series.
    Aligns on index, drops columns with all NaN.
    """
    df = pd.DataFrame(series_dict)
    df = df.dropna(how="all", axis=1)
    if df.empty:
        return pd.Series(dtype=float)
    if weights:
        w = pd.Series(weights)
        w = w[w.index.isin(df.columns)]
        w = w / w.sum()
        return df[w.index].mul(w, axis=1).sum(axis=1, min_count=1)
    return df.mean(axis=1)


def consensus_score(window_series: dict) -> pd.Series:
    """
    Consensus score = 1 - normalised std dev across window variants.
    High value (→1) = windows agree. Low value (→0) = windows diverge.
    """
    df = pd.DataFrame(window_series)
    std = df.std(axis=1)
    # Normalise: cap at 2 std units spread being max disagreement
    normalised = (std / 2).clip(0, 1)
    return (1 - normalised).rename("consensus")


# =============================================================================
# FACTOR 1: GROWTH
# Leading (60%): yield curve slope, OECD CLI, equity momentum
# Coincident (40%): GDP trend deviation
# =============================================================================

def compute_growth(df: pd.DataFrame, config) -> dict:
    """
    Returns {window_label: pd.Series} for growth factor across all windows.
    """
    components_leading   = {}
    components_coincident = {}

    # --- Leading: yield curve slope (10y-2y) ---
    if "T10Y2Y" in df.columns:
        components_leading["yield_curve"] = df["T10Y2Y"]
    else:
        logger.warning("Growth: T10Y2Y missing")

    # --- Leading: OECD CLI ---
    if "USALOLITONOSTSAM" in df.columns:
        # CLI is a level — take MoM change and z-score
        cli_mom = df["USALOLITONOSTSAM"].diff(1)
        components_leading["oecd_cli"] = cli_mom
    else:
        logger.warning("Growth: OECD CLI missing")

    # --- Leading: equity momentum (12m return on S&P 500) ---
    if "^GSPC" in df.columns:
        eq_mom = df["^GSPC"].pct_change(config.YOY_WINDOW) * 100
        components_leading["equity_momentum"] = eq_mom
    else:
        logger.warning("Growth: ^GSPC missing for equity momentum")

    # --- Coincident: GDP trend deviation ---
    if "GDP" in df.columns:
        # Log GDP trend deviation — rolling 10yr trend
        log_gdp = np.log(df["GDP"].replace(0, np.nan))
        trend   = log_gdp.rolling(config.GDP_TREND_WINDOW, min_periods=config.GDP_TREND_MIN_PERIODS).mean()
        gdp_dev = (log_gdp - trend) * 100  # in pct pts
        components_coincident["gdp_trend_dev"] = gdp_dev
    else:
        logger.warning("Growth: GDP missing")

    # --- Revision momentum (if available) ---
    revision_cols = [c for c in df.columns
                     if c.endswith("_revision") and any(
                         x in c for x in ["GDP", "USALOLITONOSTSAM"])]
    revision_signal = None
    if revision_cols:
        revision_signal = df[revision_cols].mean(axis=1)

    # --- Build composite per window ---
    results = {}
    windows = config.ZSCORE_WINDOWS
    expanding = config.ZSCORE_EXPANDING
    lw = config.GROWTH_LEAD_WEIGHT
    cw = config.GROWTH_COIN_WEIGHT

    # Z-score each sub-component across all windows first
    lead_zscores   = {}
    coin_zscores   = {}

    for name, s in components_leading.items():
        for label, zs in compute_all_windows(s.dropna(), windows, expanding).items():
            lead_zscores.setdefault(label, {})[name] = zs

    for name, s in components_coincident.items():
        for label, zs in compute_all_windows(s.dropna(), windows, expanding).items():
            coin_zscores.setdefault(label, {})[name] = zs

    all_labels = list(lead_zscores.keys() or coin_zscores.keys())

    for label in all_labels:
        lead_comp = composite_zscore(lead_zscores.get(label, {}))
        coin_comp = composite_zscore(coin_zscores.get(label, {}))

        parts = {}
        if not lead_comp.empty:
            parts["leading"] = lead_comp * lw
        if not coin_comp.empty:
            parts["coincident"] = coin_comp * cw

        if not parts:
            continue

        factor = pd.DataFrame(parts).sum(axis=1, min_count=1)

        # Add revision momentum at small weight
        if revision_signal is not None:
            rev_z = zscore_rolling(revision_signal, config.ZSCORE_MIN_PERIODS) \
                if "m" in label else zscore_expanding(revision_signal)
            factor = factor * (1 - config.REVISION_WEIGHT) + \
                     rev_z.reindex(factor.index).fillna(0) * config.REVISION_WEIGHT

        results[label] = factor.rename(f"growth_{label}")

    return results


# =============================================================================
# FACTOR 2: INFLATION
# CPI YoY, PCE YoY, 5yr5yr forward inflation, TIPS real yield (inverted),
# commodity momentum
# =============================================================================

def compute_inflation(df: pd.DataFrame, config) -> dict:
    components = {}

    # CPI YoY
    if "CPIAUCSL" in df.columns:
        cpi_yoy = df["CPIAUCSL"].pct_change(config.YOY_WINDOW) * 100
        components["cpi_yoy"] = cpi_yoy

    # PCE YoY
    if "PCEPI" in df.columns:
        pce_yoy = df["PCEPI"].pct_change(config.YOY_WINDOW) * 100
        components["pce_yoy"] = pce_yoy

    # 5yr5yr forward inflation expectation (market-implied)
    if "T5YIFR" in df.columns:
        components["fwd_inflation"] = df["T5YIFR"]

    # TIPS real yield — inverted (high real yield = disinflationary pressure)
    if "DFII5" in df.columns:
        components["tips_inverted"] = -df["DFII5"]

    # Commodity momentum (proxy for input cost inflation)
    if "DJP" in df.columns:
        com_mom = df["DJP"].pct_change(config.MOMENTUM_WINDOW_SHORT) * 100
        components["commodity_momentum"] = com_mom
    elif "GLD" in df.columns:
        gld_mom = df["GLD"].pct_change(config.MOMENTUM_WINDOW_SHORT) * 100
        components["gold_momentum"] = gld_mom

    # Revision signal
    revision_cols = [c for c in df.columns
                     if c.endswith("_revision") and any(
                         x in c for x in ["CPIAUCSL", "PCEPI"])]
    revision_signal = df[revision_cols].mean(axis=1) if revision_cols else None

    results = {}
    windows = config.ZSCORE_WINDOWS
    expanding = config.ZSCORE_EXPANDING

    window_zscores = {}
    for name, s in components.items():
        for label, zs in compute_all_windows(s.dropna(), windows, expanding).items():
            window_zscores.setdefault(label, {})[name] = zs

    for label, comp_dict in window_zscores.items():
        factor = composite_zscore(comp_dict)
        if factor.empty:
            continue

        if revision_signal is not None:
            rev_z = zscore_rolling(revision_signal, config.ZSCORE_MIN_PERIODS) \
                if "m" in label else zscore_expanding(revision_signal)
            factor = factor * (1 - config.REVISION_WEIGHT) + \
                     rev_z.reindex(factor.index).fillna(0) * config.REVISION_WEIGHT

        results[label] = factor.rename(f"inflation_{label}")

    return results


# =============================================================================
# FACTOR 3: LIQUIDITY
# Multi-central-bank aware but acknowledged thin on public data.
# Fed: M2 growth, balance sheet growth, real rate level
# USD: DXY direction (strengthening = tightening)
# Data quality flagged explicitly in output
# =============================================================================

def compute_liquidity(df: pd.DataFrame, config) -> dict:
    components = {}
    coverage_notes = []

    # Fed M2 growth YoY
    if "M2SL" in df.columns:
        m2_yoy = df["M2SL"].pct_change(config.YOY_WINDOW) * 100
        components["m2_growth"] = m2_yoy
        coverage_notes.append("Fed M2")
    else:
        logger.warning("Liquidity: M2SL missing")

    # Fed balance sheet growth YoY
    if "WALCL" in df.columns:
        fed_bs_yoy = df["WALCL"].pct_change(config.YOY_WINDOW) * 100
        components["fed_balance_sheet"] = fed_bs_yoy
        coverage_notes.append("Fed BS")
    else:
        logger.warning("Liquidity: WALCL missing")

    # Real rate = nominal yield - CPI (inverted: lower real rate = easier liquidity)
    if "T10Y2Y" in df.columns and "CPIAUCSL" in df.columns:
        cpi_yoy = df["CPIAUCSL"].pct_change(config.YOY_WINDOW) * 100
        # Use 10y yield proxy: T10Y2Y + 2y (approximate via spread)
        # For simplicity use yield curve level as real rate proxy direction
        real_rate_proxy = -(df["T10Y2Y"] - cpi_yoy.reindex(df.index))
        components["real_rate_inverted"] = real_rate_proxy
        coverage_notes.append("Real rate proxy")

    # USD direction — inverted (USD strengthening = tightening global liquidity)
    if "DTWEXBGS" in df.columns:
        usd_mom = -(df["DTWEXBGS"].pct_change(config.MOMENTUM_WINDOW_SHORT) * 100)
        components["usd_inverted"] = usd_mom
        coverage_notes.append("USD (inv)")
    else:
        logger.warning("Liquidity: DTWEXBGS missing — USD proxy unavailable")

    if not components:
        logger.error("Liquidity: no components available")
        return {}

    logger.info(f"Liquidity factor coverage: {', '.join(coverage_notes)} "
                f"(NOTE: non-Fed CB data requires Bloomberg)")

    results = {}
    windows = config.ZSCORE_WINDOWS
    expanding = config.ZSCORE_EXPANDING

    window_zscores = {}
    for name, s in components.items():
        for label, zs in compute_all_windows(s.dropna(), windows, expanding).items():
            window_zscores.setdefault(label, {})[name] = zs

    for label, comp_dict in window_zscores.items():
        factor = composite_zscore(comp_dict)
        if not factor.empty:
            results[label] = factor.rename(f"liquidity_{label}")

    return results


# =============================================================================
# FACTOR 4: RISK APPETITE
# VIX (inverted), put/call ratio (inverted — Fullerton confirmed),
# HY credit spread (inverted), EM vs DM relative performance,
# equity trend (price vs 12m moving average)
# =============================================================================

def compute_risk_appetite(df: pd.DataFrame, config) -> dict:
    components = {}

    # VIX — inverted (high VIX = low risk appetite)
    if "^VIX" in df.columns:
        components["vix_inverted"] = -df["^VIX"]
    else:
        logger.warning("Risk Appetite: VIX missing")

    # Put/call ratio — inverted (high P/C = bearish positioning = low risk appetite)
    # Fullerton explicitly confirmed as a component
    if "^PCALL" in df.columns:
        components["putcall_inverted"] = -df["^PCALL"]
    else:
        logger.warning("Risk Appetite: Put/call ratio missing — "
                       "check ^PCALL availability in Yahoo Finance")

    # HY credit spread — inverted (wide spread = low risk appetite)
    if "BAMLH0A0HYM2" in df.columns:
        components["hy_spread_inverted"] = -df["BAMLH0A0HYM2"]
    else:
        logger.warning("Risk Appetite: HY spread missing")

    # EM vs DM relative performance (EEM / VT ratio momentum)
    if "EEM" in df.columns and "VT" in df.columns:
        em_dm_ratio = df["EEM"] / df["VT"].replace(0, np.nan)
        em_dm_mom   = em_dm_ratio.pct_change(config.MOMENTUM_WINDOW_SHORT) * 100
        components["em_dm_relative"] = em_dm_mom

    # Equity trend: S&P 500 vs its 12m moving average
    if "^GSPC" in df.columns:
        ma12 = df["^GSPC"].rolling(config.EQUITY_MA_WINDOW, min_periods=config.EQUITY_MA_WINDOW // 2).mean()
        eq_trend = (df["^GSPC"] / ma12.replace(0, np.nan) - 1) * 100
        components["equity_trend"] = eq_trend

    # HYG/LQD ratio momentum — credit risk appetite signal
    if "HYG" in df.columns and "LQD" in df.columns:
        hyg_lqd = df["HYG"] / df["LQD"].replace(0, np.nan)
        hyg_lqd_mom = hyg_lqd.pct_change(config.MOMENTUM_WINDOW_SHORT) * 100
        components["hyg_lqd_ratio"] = hyg_lqd_mom

    if not components:
        logger.error("Risk Appetite: no components available")
        return {}

    results = {}
    windows = config.ZSCORE_WINDOWS
    expanding = config.ZSCORE_EXPANDING

    window_zscores = {}
    for name, s in components.items():
        for label, zs in compute_all_windows(s.dropna(), windows, expanding).items():
            window_zscores.setdefault(label, {})[name] = zs

    for label, comp_dict in window_zscores.items():
        factor = composite_zscore(comp_dict)
        if not factor.empty:
            results[label] = factor.rename(f"risk_appetite_{label}")

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def compute_factors(df: pd.DataFrame, config) -> tuple[pd.DataFrame, dict]:
    """
    Compute all four factors across all z-score windows.

    Parameters
    ----------
    df      : output of vintage_manager.build_monthly_frame()
    config  : config module

    Returns
    -------
    factors_df  : pd.DataFrame
                  Columns: growth_36m, growth_60m, ... inflation_36m, ...
                           liquidity_36m, ... risk_appetite_36m, ...
                  Plus: growth_consensus, inflation_consensus, etc.
    meta        : dict — coverage notes, missing components, window list
    """
    logger.info("Computing factors...")

    factor_results = {
        "growth":        compute_growth(df, config),
        "inflation":     compute_inflation(df, config),
        "liquidity":     compute_liquidity(df, config),
        "risk_appetite": compute_risk_appetite(df, config),
    }

    # Assemble wide dataframe
    all_series = {}
    consensus_series = {}
    meta = {"windows": config.ZSCORE_WINDOWS, "expanding": config.ZSCORE_EXPANDING,
            "factors": {}}

    for factor_name, window_dict in factor_results.items():
        if not window_dict:
            logger.warning(f"Factor {factor_name}: no output — skipping")
            meta["factors"][factor_name] = {"status": "missing"}
            continue

        meta["factors"][factor_name] = {
            "status":  "ok",
            "windows": list(window_dict.keys()),
        }

        for label, s in window_dict.items():
            col = f"{factor_name}_{label}"
            all_series[col] = s

        # Consensus score across windows for this factor
        cons = consensus_score(window_dict)
        consensus_series[f"{factor_name}_consensus"] = cons

    if not all_series:
        logger.error("Factor engine: no output produced")
        return pd.DataFrame(), meta

    factors_df = pd.DataFrame(all_series)

    # Add consensus scores
    for col, s in consensus_series.items():
        factors_df[col] = s.reindex(factors_df.index)

    factors_df = factors_df.sort_index()

    # Log summary
    n_rows = len(factors_df.dropna(how="all"))
    logger.info(
        f"Factor engine complete. {len(factors_df.columns)} columns, "
        f"{n_rows} months with data. "
        f"Date range: {factors_df.index[0].date()} to {factors_df.index[-1].date()}"
    )

    return factors_df, meta


def get_current_factors(factors_df: pd.DataFrame) -> pd.Series:
    """
    Return the most recent row of factor scores as a named Series.
    Drops consensus columns — returns only factor_window scores.
    """
    factor_cols = [c for c in factors_df.columns if not c.endswith("_consensus")]
    latest = factors_df[factor_cols].dropna(how="all").iloc[-1]
    return latest


def get_factor_summary(factors_df: pd.DataFrame, config) -> pd.DataFrame:
    """
    For each factor, return current score across all windows plus consensus.
    Useful for dashboard current-state panel.

    Returns DataFrame with index = factor name,
    columns = [36m, 60m, 120m, 240m, expanding, consensus]
    """
    factor_names = ["growth", "inflation", "liquidity", "risk_appetite"]
    rows = []

    for fname in factor_names:
        row = {"factor": fname}
        for w in config.ZSCORE_WINDOWS:
            col = f"{fname}_{w}m"
            if col in factors_df.columns:
                vals = factors_df[col].dropna()
                row[f"{w}m"] = round(vals.iloc[-1], 3) if not vals.empty else np.nan

        if config.ZSCORE_EXPANDING:
            col = f"{fname}_expanding"
            if col in factors_df.columns:
                vals = factors_df[col].dropna()
                row["expanding"] = round(vals.iloc[-1], 3) if not vals.empty else np.nan

        cons_col = f"{fname}_consensus"
        if cons_col in factors_df.columns:
            vals = factors_df[cons_col].dropna()
            row["consensus"] = round(vals.iloc[-1], 3) if not vals.empty else np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("factor")
