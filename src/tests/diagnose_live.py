# =============================================================================
# LIVE DATA DIAGNOSTIC
# Run this when live data shows None factors to identify which series
# are failing and why.
# Run from C:\dev\taa with: python src/tests/diagnose_live.py
# =============================================================================

import sys, logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

SRC = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

import config

def divider(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


divider("1. RAW FETCH — FRED")
from data.fetcher import _get_fred_data

fred_series = {
    "GDP":             "GDP",
    "OECD CLI":        config.FRED_OECD_CLI_ID,
    "T10Y2Y":          "T10Y2Y",
    "CPI":             "CPIAUCSL",
    "PCE":             "PCEPI",
    "T5YIFR":          "T5YIFR",
    "DFII5":           "DFII5",
    "M2":              "M2SL",
    "Fed Balance":     "WALCL",
    "USD Index":       "DTWEXBGS",
    "HY Spread":       "BAMLH0A0HYM2",
}

fred_results = {}
for label, sid in fred_series.items():
    try:
        s = _get_fred_data(sid, config.HISTORY_START)
        if s.empty:
            print(f"  EMPTY   {label} ({sid})")
        else:
            print(f"  OK      {label} ({sid})  "
                  f"{len(s)} obs  {s.index[0].date()} -> {s.index[-1].date()}  "
                  f"latest={s.iloc[-1]:.3f}")
            fred_results[sid] = s
    except Exception as e:
        print(f"  ERROR   {label} ({sid}): {e}")

divider("2. RAW FETCH — YAHOO")
from data.fetcher import _get_yahoo_data

yahoo_tickers = {
    "VIX":       "^VIX",
    "SPX":       "^GSPC",
    # Note: ^PCALL not available on Yahoo Finance
    "VT":        "VT",
    "EEM":       "EEM",
    "HYG":       "HYG",
    "LQD":       "LQD",
    "TLT":       "TLT",
    "AGG":       "AGG",
    "DJP":       "DJP",
    "GLD":       "GLD",
}

for label, ticker in yahoo_tickers.items():
    try:
        s = _get_yahoo_data(ticker, "2020-01-01")
        if s.empty:
            print(f"  EMPTY   {label} ({ticker})")
        else:
            print(f"  OK      {label} ({ticker})  "
                  f"{len(s)} obs  latest={s.iloc[-1]:.2f} on {s.index[-1].date()}")
    except Exception as e:
        print(f"  ERROR   {label} ({ticker}): {e}")

divider("3. VINTAGE MANAGER")
from data.fetcher import get_data
from data.vintage_manager import build_monthly_frame

print("Fetching all data...")
raw = get_data(config, force_refresh=True)
print(f"Series fetched: {len(raw)}")

df, quality = build_monthly_frame(raw, config)
print(f"Monthly frame: {df.shape}")
print()

base_cols = [c for c in df.columns
             if not c.endswith("_revision") and not c.endswith("_surprise")]
print("Last 3 values per series (None = missing):")
for col in base_cols:
    s = df[col].dropna()
    if s.empty:
        print(f"  EMPTY   {col}")
    else:
        recent = s.tail(3).values
        print(f"  {col:<25} {recent}")

divider("4. FACTOR ENGINE — SUB-COMPONENT CHECK")
from factors.engine import compute_factors

factors_df, meta = compute_factors(df, config)

factor_names = ["growth", "inflation", "liquidity", "risk_appetite"]
windows = [f"{w}m" for w in config.ZSCORE_WINDOWS]

print("Factor scores (60m window, last 3 months):")
for fname in factor_names:
    col = f"{fname}_60m"
    if col not in factors_df.columns:
        print(f"  MISSING  {fname}")
        continue
    s = factors_df[col].dropna()
    if s.empty:
        print(f"  EMPTY    {fname}")
    else:
        print(f"  {fname:<16} last={s.iloc[-1]:+.3f}  "
              f"n_valid={len(s)}  "
              f"last_date={s.index[-1].date()}")

divider("5. GROWTH FACTOR SUB-COMPONENTS")
# Check each input to growth factor individually
checks = {
    "T10Y2Y (yield curve)": ("T10Y2Y", None),
    "OECD CLI MoM":         (config.FRED_OECD_CLI_ID, "diff"),
    "SPX 12m momentum":     ("^GSPC", "pct12"),
    "GDP log deviation":    ("GDP", "log_dev"),
}
for label, (col, transform) in checks.items():
    if col not in df.columns:
        print(f"  MISSING  {label} — {col} not in monthly frame")
        continue
    s = df[col].dropna()
    if s.empty:
        print(f"  EMPTY    {label}")
        continue
    if transform == "diff":
        s = s.diff(1).dropna()
    elif transform == "pct12":
        s = s.pct_change(12).dropna()
    elif transform == "log_dev":
        import numpy as np
        log_s = np.log(s.replace(0, np.nan))
        trend = log_s.rolling(120, min_periods=24).mean()
        s = (log_s - trend).dropna()
    print(f"  OK       {label}  n={len(s)}  latest={s.iloc[-1]:+.4f}")

divider("6. LIQUIDITY FACTOR SUB-COMPONENTS")
liq_checks = {
    "M2 YoY":           ("M2SL", "pct12"),
    "Fed BS YoY":       ("WALCL", "pct12"),
    "USD 3m momentum":  ("DTWEXBGS", "pct3"),
    "T10Y2Y (real rt)": ("T10Y2Y", None),
    "CPI YoY":          ("CPIAUCSL", "pct12"),
}
for label, (col, transform) in liq_checks.items():
    if col not in df.columns:
        print(f"  MISSING  {label} — {col} not in monthly frame")
        continue
    s = df[col].dropna()
    if s.empty:
        print(f"  EMPTY    {label}")
        continue
    if transform == "pct12":
        s = s.pct_change(12).dropna()
    elif transform == "pct3":
        s = s.pct_change(3).dropna()
    print(f"  OK       {label}  n={len(s)}  latest={s.iloc[-1]:+.4f}")

print()
print("=" * 60)
print("  DIAGNOSTIC COMPLETE")
print("  Share this output to identify the root cause.")
print("=" * 60)
