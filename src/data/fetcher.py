# =============================================================================
# MODULE 1a: DATA FETCHER
# Pulls raw data from FRED and Yahoo Finance.
# Outputs a clean wide dataframe with period_date and as_of_date per series.
# Nothing is computed here — just ingestion and standardisation.
# =============================================================================

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _get_fred_data(series_id: str, start: str) -> pd.Series:
    """Pull a single series from FRED via pandas_datareader."""
    try:
        import pandas_datareader.data as web
        s = web.DataReader(series_id, "fred", start=start)
        return s.squeeze()
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
        return pd.Series(dtype=float, name=series_id)


def _get_yahoo_data(ticker: str, start: str) -> pd.Series:
    """Pull adjusted close price from Yahoo Finance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"Empty response for {ticker}")
        return df["Close"].squeeze()
    except Exception as e:
        logger.warning(f"Yahoo fetch failed for {ticker}: {e}")
        return pd.Series(dtype=float, name=ticker)


def fetch_all(config) -> dict:
    """
    Fetch all raw series defined in config.
    Returns a dict:
      {
        series_id: {
          "data":     pd.Series (datetime index, float values),
          "source":   "FRED" | "YAHOO",
          "fetched_at": datetime (UTC now — the as_of timestamp for live data),
          "frequency": "D" | "M" | "Q"
        }
      }
    All series returned at native frequency.
    Resampling to monthly happens in vintage_manager.
    """
    fetched_at = datetime.utcnow()
    results = {}

    # --- FRED series ---------------------------------------------------------
    fred_ids = list(config.FRED_SERIES.keys())
    # Replace OECD_CLI placeholder with actual FRED id
    fred_ids = [
        config.FRED_OECD_CLI_ID if s == "OECD_CLI" else s
        for s in fred_ids
    ]
    fred_ids = list(dict.fromkeys(fred_ids))  # deduplicate, preserve order

    for sid in fred_ids:
        logger.info(f"Fetching FRED: {sid}")
        s = _get_fred_data(sid, config.HISTORY_START)
        if not s.empty:
            results[sid] = {
                "data":       s,
                "source":     "FRED",
                "fetched_at": fetched_at,
                "frequency":  _infer_frequency(s),
            }

    # --- Yahoo tickers -------------------------------------------------------
    for ticker, desc in config.YAHOO_TICKERS.items():
        logger.info(f"Fetching Yahoo: {ticker} — {desc}")
        s = _get_yahoo_data(ticker, config.HISTORY_START)
        if not s.empty:
            results[ticker] = {
                "data":       s,
                "source":     "YAHOO",
                "fetched_at": fetched_at,
                "frequency":  _infer_frequency(s),
            }

    logger.info(f"Fetcher complete. {len(results)} series loaded.")
    return results


def _infer_frequency(s: pd.Series) -> str:
    """Infer D / M / Q from median gap between observations."""
    if len(s) < 2:
        return "D"
    gaps = pd.Series(s.index).diff().dropna().dt.days
    median_gap = gaps.median()
    if median_gap <= 8:
        return "D"
    elif median_gap <= 35:
        return "M"
    else:
        return "Q"


def load_cache(cache_path: Path) -> dict | None:
    """Load previously cached raw data from parquet files."""
    if not cache_path.exists():
        return None
    try:
        results = {}
        meta_path = cache_path / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        for sid, info in meta.items():
            pq = cache_path / f"{sid.replace('^','_')}.parquet"
            if pq.exists():
                s = pd.read_parquet(pq).squeeze()
                results[sid] = {
                    "data":       s,
                    "source":     info["source"],
                    "fetched_at": datetime.fromisoformat(info["fetched_at"]),
                    "frequency":  info["frequency"],
                }
        logger.info(f"Cache loaded. {len(results)} series.")
        return results
    except Exception as e:
        logger.warning(f"Cache load failed: {e}")
        return None


def save_cache(results: dict, cache_path: Path):
    """Save raw fetched data to parquet for fast reload."""
    cache_path.mkdir(parents=True, exist_ok=True)
    meta = {}
    for sid, info in results.items():
        safe = sid.replace("^", "_")
        info["data"].to_frame(name=sid).to_parquet(cache_path / f"{safe}.parquet")
        meta[sid] = {
            "source":     info["source"],
            "fetched_at": info["fetched_at"].isoformat(),
            "frequency":  info["frequency"],
        }
    with open(cache_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Cache saved to {cache_path}")


def get_data(config, force_refresh: bool = False, use_snapshot: bool = False) -> dict:
    """
    Main entry point. Returns raw series dict.

    Priority order:
      1. Local cache (if fresh and not force_refresh)
      2. Live fetch from FRED + Yahoo
      3. Committed snapshot (fallback for Streamlit Cloud or offline use)
      4. Empty dict (all downstream code handles this gracefully)

    Parameters
    ----------
    force_refresh  : bypass cache and fetch live
    use_snapshot   : skip live fetch and load committed snapshot directly
    """
    from data.snapshot import load_snapshot

    cache_path = config.DATA_DIR

    # --- Snapshot mode (Streamlit Cloud showcase) ----------------------------
    if use_snapshot:
        logger.info("Snapshot mode — loading committed data snapshot.")
        snap = load_snapshot()
        if snap:
            return snap
        logger.warning("Snapshot not found — falling back to live fetch.")

    # --- Check local cache ---------------------------------------------------
    if not force_refresh:
        cached = load_cache(cache_path)
        if cached:
            sample = next(iter(cached.values()))
            age_hours = (datetime.utcnow() - sample["fetched_at"]).total_seconds() / 3600
            if age_hours < config.REFRESH_CACHE_HOURS:
                logger.info(f"Cache is {age_hours:.1f}h old — using cached data.")
                return cached
            else:
                logger.info(f"Cache is {age_hours:.1f}h old — refreshing.")

    # --- Live fetch ----------------------------------------------------------
    try:
        results = fetch_all(config)
        if results:
            save_cache(results, cache_path)
            return results
    except Exception as e:
        logger.warning(f"Live fetch failed: {e} — trying snapshot fallback.")

    # --- Snapshot fallback ---------------------------------------------------
    snap = load_snapshot()
    if snap:
        logger.info("Using committed snapshot as fallback.")
        return snap

    logger.error("No data available — cache empty, live fetch failed, no snapshot.")
    return {}
