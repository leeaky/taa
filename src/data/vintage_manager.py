# =============================================================================
# MODULE 1b: VINTAGE MANAGER
# Sits between fetcher and factor engine.
# Responsibilities:
#   1. Resample all series to monthly frequency
#   2. Apply publication lag offsets → as_of_date adjusted data
#   3. Compute revision deltas and release surprises
#   4. Flag staleness per series
#   5. Output a single clean monthly dataframe + data quality dict
# =============================================================================

import logging
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Minimum history (months) needed before a series is considered usable
MIN_HISTORY_MONTHS = 36


def resample_to_monthly(s: pd.Series, frequency: str) -> pd.Series:
    """
    Resample a series to month-end frequency.
    Daily → last observation of month (prices) or mean (rates/spreads).
    Monthly → already monthly, just align to month-end.
    Quarterly → forward-fill to monthly.
    """
    if s.empty:
        return s

    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()

    if frequency == "D":
        # Use last observation of each month for prices/rates
        return s.resample("ME").last().dropna()
    elif frequency == "M":
        return s.resample("ME").last().dropna()
    elif frequency == "Q":
        # Forward-fill quarterly data to monthly
        monthly_idx = pd.date_range(s.index[0], s.index[-1], freq="ME")
        return s.reindex(monthly_idx).ffill().dropna()
    else:
        return s.resample("ME").last().dropna()


def apply_lag(s: pd.Series, lag_days: int) -> pd.Series:
    """
    Shift the index forward by lag_days to reflect when data was actually
    published. A CPI reading for October, published on Nov 13, gets
    as_of_date = Nov 13 (approximately month-end + 45 days).

    We work in monthly space so we convert lag_days to months (ceiling).
    This means the data point is not available until that month-end.
    """
    lag_months = int(np.ceil(lag_days / 30))
    return s.shift(lag_months)


def compute_revision_delta(s: pd.Series, window: int = 3) -> pd.Series:
    """
    Proxy for revision signal: rolling change in the series itself.
    In absence of true vintage data, the direction and magnitude of
    recent changes serves as a revision/momentum signal.
    Normalised to z-score over the same window.
    """
    delta = s.diff(1)
    rolling_mean = delta.rolling(window).mean()
    rolling_std  = delta.rolling(window).std()
    z = (delta - rolling_mean) / rolling_std.replace(0, np.nan)
    return z.rename(f"{s.name}_revision")


def compute_release_surprise(s: pd.Series) -> pd.Series:
    """
    Surprise = current release vs prior reading (naive expectation).
    Normalised. In a professional model this would use consensus estimates.
    """
    surprise = s.diff(1) / s.shift(1).abs().replace(0, np.nan)
    return surprise.rename(f"{s.name}_surprise")


def check_staleness(sid: str, s: pd.Series, lag_days: int, as_of: datetime) -> dict:
    """
    Check whether the series has a recent enough observation given its
    expected publication lag. Returns a staleness dict.
    """
    if s.empty:
        return {"stale": True, "reason": "empty series", "last_period": None, "tier": "UNKNOWN"}

    last_period = s.index[-1]
    expected_available = last_period + timedelta(days=lag_days)
    days_since_expected = (as_of - expected_available).days

    # A series is stale if the most recent expected release hasn't arrived
    # Allow 5 business days grace
    stale = days_since_expected < -7

    return {
        "stale":             stale,
        "last_period":       last_period.strftime("%Y-%m"),
        "expected_available": expected_available.strftime("%Y-%m-%d"),
        "days_overdue":      max(0, -days_since_expected) if stale else 0,
        "reason":            "awaiting publication" if stale else "current",
    }


def build_monthly_frame(raw: dict, config, as_of: datetime = None) -> tuple[pd.DataFrame, dict]:
    """
    Main entry point for vintage manager.

    Parameters
    ----------
    raw     : output from fetcher.get_data()
    config  : config module
    as_of   : datetime — treat this as 'today'. Defaults to utcnow.

    Returns
    -------
    df          : pd.DataFrame, monthly frequency, lag-adjusted
                  Columns: series values (lag-adjusted)
                  Plus _revision and _surprise columns per macro series
    quality     : dict — staleness and tier info per series
    """
    if as_of is None:
        as_of = datetime.utcnow()

    monthly = {}
    revisions = {}
    surprises = {}
    quality = {}

    lag_map = config.SERIES_LAGS_DAYS
    tier_map = config.DATA_TIERS

    # Map OECD_CLI config key to actual FRED id
    oecd_key = config.FRED_OECD_CLI_ID

    for sid, info in raw.items():
        s = info["data"].copy()
        freq = info["frequency"]

        # Normalise index to datetime
        s.index = pd.to_datetime(s.index)
        s.name = sid

        # Drop future dates (shouldn't happen but defensive)
        s = s[s.index <= pd.Timestamp(as_of)]

        if s.empty or len(s) < 3:
            logger.warning(f"Skipping {sid} — insufficient data.")
            quality[sid] = {"stale": True, "reason": "insufficient data"}
            continue

        # Resample to monthly
        s_monthly = resample_to_monthly(s, freq)

        # Apply publication lag (shifts data forward in time)
        lag_days = lag_map.get(sid, 0)
        s_lagged = apply_lag(s_monthly, lag_days)

        # Clip to as_of date (no look-ahead)
        s_lagged = s_lagged[s_lagged.index <= pd.Timestamp(as_of)]

        if s_lagged.empty:
            logger.warning(f"Skipping {sid} — empty after lag adjustment.")
            continue

        monthly[sid] = s_lagged

        # Revision and surprise signals for macro series only (not market prices)
        if info["source"] == "FRED":
            revisions[sid] = compute_revision_delta(s_lagged)
            surprises[sid]  = compute_release_surprise(s_lagged)

        # Staleness check
        stale_info = check_staleness(sid, s_monthly, lag_days, as_of)
        stale_info["tier"]   = tier_map.get(sid, "REVISED")
        stale_info["source"] = info["source"]
        quality[sid] = stale_info

        if stale_info["stale"]:
            logger.warning(f"STALE: {sid} — {stale_info['reason']}")

    if not monthly:
        logger.error("No series available after vintage processing.")
        return pd.DataFrame(), quality

    # Combine into single dataframe
    df = pd.DataFrame(monthly)

    # Add revision and surprise columns
    for sid, rev in revisions.items():
        col = f"{sid}_revision"
        df[col] = rev.reindex(df.index)

    for sid, sur in surprises.items():
        col = f"{sid}_surprise"
        df[col] = sur.reindex(df.index)

    # Sort index
    df = df.sort_index()

    # Log data quality summary
    n_stale = sum(1 for v in quality.values() if v.get("stale"))
    n_revised = sum(1 for v in quality.values() if v.get("tier") == "REVISED")
    logger.info(
        f"Vintage manager complete. "
        f"{len(df.columns)} columns, {len(df)} months. "
        f"Stale series: {n_stale}. Revised-data series: {n_revised}."
    )

    return df, quality


def get_data_quality_summary(quality: dict) -> pd.DataFrame:
    """
    Returns a tidy dataframe summarising data quality for dashboard display.
    """
    rows = []
    for sid, info in quality.items():
        rows.append({
            "series":    sid,
            "stale":     info.get("stale", True),
            "tier":      info.get("tier", "UNKNOWN"),
            "last_period": info.get("last_period", "—"),
            "reason":    info.get("reason", "—"),
        })
    return pd.DataFrame(rows).set_index("series")
