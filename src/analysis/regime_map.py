# =============================================================================
# MODULE 4: HISTORICAL REGIME MAP
# For each historical regime period, shows what asset classes actually returned.
# This is descriptive, not predictive — a historical map, not a backtest.
#
# Two data sources, clearly labelled:
#   BOOTSTRAP  — built from revised public data before state log exists
#   LIVE       — built from the state log (epistemically honest)
#
# Outputs:
#   regime_periods    — start/end/duration of each regime period
#   asset_stats       — median return, hit rate, max drawdown by regime × asset
#   summary_matrix    — clean regime × asset heatmap for display
#   return_series     — full monthly return series per asset per regime period
# =============================================================================

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# RETURN CALCULATIONS
# =============================================================================

def compute_monthly_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price levels to monthly total returns (%).
    Input: wide DataFrame of monthly prices.
    Output: wide DataFrame of monthly returns (%).
    """
    return prices_df.pct_change(1) * 100


def compute_period_return(returns: pd.Series, start: pd.Timestamp,
                          end: pd.Timestamp) -> float:
    """
    Compound return over a period. Returns NaN if insufficient data.
    """
    period = returns.loc[start:end].dropna()
    if len(period) < 1:
        return np.nan
    cumulative = (1 + period / 100).prod() - 1
    return round(cumulative * 100, 2)


def compute_max_drawdown(returns: pd.Series, start: pd.Timestamp,
                         end: pd.Timestamp) -> float:
    """
    Maximum drawdown (%) over a period from monthly returns.
    """
    period = returns.loc[start:end].dropna()
    if len(period) < 2:
        return np.nan
    cumulative = (1 + period / 100).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max * 100
    return round(drawdown.min(), 2)


def compute_annualised_return(returns: pd.Series, start: pd.Timestamp,
                               end: pd.Timestamp) -> float:
    """
    Annualised return (%) over a period.
    """
    period = returns.loc[start:end].dropna()
    n_months = len(period)
    if n_months < 1:
        return np.nan
    cumulative = (1 + period / 100).prod()
    annualised = cumulative ** (12 / n_months) - 1
    return round(annualised * 100, 2)


# =============================================================================
# REGIME PERIOD EXTRACTION
# =============================================================================

def extract_regime_periods(regime_series: pd.Series) -> pd.DataFrame:
    """
    Convert a monthly regime label series into a table of contiguous periods.

    Returns DataFrame with columns:
        regime, start, end, duration_months
    """
    if regime_series.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "duration_months"])

    periods = []
    current_regime = None
    current_start  = None

    for dt, regime in regime_series.items():
        if regime is None or str(regime) in ("nan", "None"):
            continue
        if regime != current_regime:
            if current_regime is not None:
                periods.append({
                    "regime":          current_regime,
                    "start":           current_start,
                    "end":             dt,
                    "duration_months": len(regime_series.loc[current_start:dt]) - 1,
                })
            current_regime = regime
            current_start  = dt

    # Close final period
    if current_regime is not None:
        end = regime_series.index[-1]
        periods.append({
            "regime":          current_regime,
            "start":           current_start,
            "end":             end,
            "duration_months": len(regime_series.loc[current_start:end]),
        })

    df = pd.DataFrame(periods)
    if not df.empty:
        df["start"] = pd.to_datetime(df["start"])
        df["end"]   = pd.to_datetime(df["end"])
    return df


# =============================================================================
# ASSET STATISTICS BY REGIME
# =============================================================================

def compute_asset_stats(
    regime_periods: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    assets: list,
) -> pd.DataFrame:
    """
    For each regime period and each asset, compute return statistics.
    Vectorised — builds return slices per period using boolean indexing.
    """
    if regime_periods.empty or monthly_returns.empty:
        return pd.DataFrame()

    rows = []
    ret_index = monthly_returns.index

    for _, period in regime_periods.iterrows():
        regime   = period["regime"]
        start    = period["start"]
        end      = period["end"]
        n_months = period["duration_months"]

        # Slice once per period, reuse for all assets
        mask   = (ret_index >= start) & (ret_index <= end)
        slice_ = monthly_returns.loc[mask, [a for a in assets
                                            if a in monthly_returns.columns]]

        if slice_.empty:
            continue

        for asset in assets:
            if asset not in slice_.columns:
                continue

            s = slice_[asset].dropna()
            if s.empty:
                continue

            n = len(s)
            cumulative  = (1 + s / 100).prod() - 1
            total_ret   = round(cumulative * 100, 2)
            ann_ret     = round((((1 + cumulative) ** (12 / n)) - 1) * 100, 2) if n >= 1 else np.nan
            cum_series  = (1 + s / 100).cumprod()
            roll_max    = cum_series.cummax()
            max_dd      = round(((cum_series - roll_max) / roll_max * 100).min(), 2) \
                          if n >= 2 else np.nan

            rows.append({
                "regime":            regime,
                "asset":             asset,
                "start":             start,
                "end":               end,
                "total_return":      total_ret,
                "annualised_return": ann_ret,
                "max_drawdown":      max_dd,
                "n_months":          n_months,
            })

    return pd.DataFrame(rows)


def compute_regime_asset_summary(asset_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-period stats into summary statistics by regime × asset.

    Returns DataFrame with columns:
        regime, asset,
        median_total_return, mean_total_return,
        median_annualised_return,
        hit_rate (% of periods with positive return),
        median_max_drawdown,
        n_periods, avg_duration_months
    """
    if asset_stats.empty:
        return pd.DataFrame()

    rows = []
    for (regime, asset), grp in asset_stats.groupby(["regime", "asset"]):
        returns = grp["total_return"].dropna()
        if returns.empty:
            continue

        rows.append({
            "regime":                   regime,
            "asset":                    asset,
            "median_total_return":      round(returns.median(), 2),
            "mean_total_return":        round(returns.mean(), 2),
            "median_annualised_return": round(grp["annualised_return"].dropna().median(), 2),
            "hit_rate":                 round((returns > 0).mean() * 100, 1),
            "median_max_drawdown":      round(grp["max_drawdown"].dropna().median(), 2),
            "n_periods":                len(grp),
            "avg_duration_months":      round(grp["n_months"].mean(), 1),
        })

    return pd.DataFrame(rows)


def build_summary_matrix(
    summary: pd.DataFrame,
    metric: str = "median_annualised_return",
    regimes: Optional[list] = None,
    assets: Optional[list] = None,
) -> pd.DataFrame:
    """
    Pivot summary into a regime × asset matrix for heatmap display.
    Default metric: median annualised return.

    Parameters
    ----------
    metric : column from compute_regime_asset_summary to use as values
    regimes : ordered list of regime names (rows)
    assets  : ordered list of asset names (columns)
    """
    if summary.empty:
        return pd.DataFrame()

    matrix = summary.pivot_table(
        index="regime", columns="asset", values=metric, aggfunc="first"
    )

    if regimes:
        matrix = matrix.reindex([r for r in regimes if r in matrix.index])
    if assets:
        matrix = matrix.reindex(
            [a for a in assets if a in matrix.columns], axis=1
        )

    return matrix


# =============================================================================
# DATA SOURCE LABELLING
# =============================================================================

def label_data_source(regime_periods: pd.DataFrame,
                       state_log_start: Optional[pd.Timestamp]) -> pd.DataFrame:
    """
    Tag each regime period as BOOTSTRAP or LIVE based on whether
    it occurred before or after the state log started.

    BOOTSTRAP = revised public data, illustrative
    LIVE      = from state log, epistemically honest
    """
    df = regime_periods.copy()
    if state_log_start is None:
        df["data_source"] = "BOOTSTRAP"
    else:
        df["data_source"] = df["start"].apply(
            lambda s: "LIVE" if s >= state_log_start else "BOOTSTRAP"
        )
    return df


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def build_regime_map(
    ensemble_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    config,
    state_log_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Full regime map pipeline.

    Parameters
    ----------
    ensemble_df   : output of classifier.classify_regimes() — has regime_primary column
    prices_df     : monthly prices for all asset classes (from vintage_manager)
    config        : config module
    state_log_df  : optional — state log dataframe; if provided, periods are tagged LIVE/BOOTSTRAP

    Returns
    -------
    dict with keys:
        regime_periods   : pd.DataFrame — one row per contiguous regime period
        asset_stats      : pd.DataFrame — per-period stats (long format)
        summary          : pd.DataFrame — aggregated stats by regime × asset
        matrix_return    : pd.DataFrame — regime × asset median annualised return matrix
        matrix_hitrate   : pd.DataFrame — regime × asset hit rate matrix
        matrix_drawdown  : pd.DataFrame — regime × asset median max drawdown matrix
        monthly_returns  : pd.DataFrame — monthly asset returns used
        data_note        : str — transparency note about data sources
    """
    logger.info("Building regime map...")

    # --- Asset returns -------------------------------------------------------
    assets = [a for a in config.REGIME_MAP_ASSETS if a in prices_df.columns]
    if not assets:
        logger.error("Regime map: no asset price data available")
        return {}

    missing = [a for a in config.REGIME_MAP_ASSETS if a not in prices_df.columns]
    if missing:
        logger.warning(f"Regime map: assets missing from price data: {missing}")

    monthly_returns = compute_monthly_returns(prices_df[assets])

    # --- Regime periods -------------------------------------------------------
    regime_series = ensemble_df["regime_primary"].dropna()
    regime_periods = extract_regime_periods(regime_series)

    if regime_periods.empty:
        logger.error("Regime map: no regime periods extracted")
        return {}

    # Tag data source
    state_log_start = None
    if state_log_df is not None and not state_log_df.empty:
        if "run_timestamp" in state_log_df.columns:
            state_log_start = pd.to_datetime(
                state_log_df["run_timestamp"].min()
            )
    regime_periods = label_data_source(regime_periods, state_log_start)

    # --- Statistics ----------------------------------------------------------
    asset_stats = compute_asset_stats(regime_periods, monthly_returns, assets)
    summary     = compute_regime_asset_summary(asset_stats)

    regime_order = list(config.REGIME_LABELS.keys())

    matrix_return   = build_summary_matrix(summary, "median_annualised_return",
                                            regime_order, assets)
    matrix_hitrate  = build_summary_matrix(summary, "hit_rate",
                                            regime_order, assets)
    matrix_drawdown = build_summary_matrix(summary, "median_max_drawdown",
                                            regime_order, assets)

    # --- Data note -----------------------------------------------------------
    n_bootstrap = (regime_periods["data_source"] == "BOOTSTRAP").sum()
    n_live      = (regime_periods["data_source"] == "LIVE").sum()
    data_note = (
        f"{len(regime_periods)} regime periods total. "
        f"{n_bootstrap} BOOTSTRAP (revised public data — illustrative). "
        f"{n_live} LIVE (state log — epistemically honest). "
        f"Asset returns from monthly ETF prices."
    )

    logger.info(
        f"Regime map complete. {len(regime_periods)} periods, "
        f"{len(assets)} assets. {data_note}"
    )

    return {
        "regime_periods":  regime_periods,
        "asset_stats":     asset_stats,
        "summary":         summary,
        "matrix_return":   matrix_return,
        "matrix_hitrate":  matrix_hitrate,
        "matrix_drawdown": matrix_drawdown,
        "monthly_returns": monthly_returns,
        "data_note":       data_note,
    }


# =============================================================================
# CONVENIENCE ACCESSORS FOR DASHBOARD
# =============================================================================

def get_regime_summary_for_display(
    regime_map: dict,
    regime_name: str,
) -> dict:
    """
    Return all stats for a single regime — for regime detail panel in dashboard.
    """
    summary = regime_map.get("summary", pd.DataFrame())
    periods = regime_map.get("regime_periods", pd.DataFrame())

    if summary.empty:
        return {}

    regime_summary = summary[summary["regime"] == regime_name]
    regime_periods = periods[periods["regime"] == regime_name]

    return {
        "asset_stats":   regime_summary.set_index("asset").to_dict("index"),
        "n_periods":     len(regime_periods),
        "avg_duration":  round(regime_periods["duration_months"].mean(), 1)
                         if not regime_periods.empty else None,
        "total_months":  regime_periods["duration_months"].sum()
                         if not regime_periods.empty else 0,
        "periods":       regime_periods.to_dict("records"),
    }


def get_current_regime_historical_context(
    regime_map: dict,
    current_regime: str,
    config,
) -> dict:
    """
    For the current regime, return a quick summary of how assets have
    historically performed — for the dashboard current state panel.
    """
    summary = regime_map.get("summary", pd.DataFrame())
    if summary.empty or not current_regime:
        return {}

    subset = summary[summary["regime"] == current_regime].copy()
    if subset.empty:
        return {}

    subset = subset.sort_values("median_annualised_return", ascending=False)

    return {
        "regime":        current_regime,
        "best_asset":    subset.iloc[0]["asset"] if not subset.empty else None,
        "worst_asset":   subset.iloc[-1]["asset"] if len(subset) > 1 else None,
        "asset_returns": subset.set_index("asset")[
            ["median_annualised_return", "hit_rate", "median_max_drawdown"]
        ].to_dict("index"),
    }
