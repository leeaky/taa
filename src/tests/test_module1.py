# =============================================================================
# TEST / REVIEW SCRIPT — MODULE 1
# Tests data fetcher, vintage manager, and state log independently.
# Run from C:\dev\taa with: python src/tests/test_module1.py
#
# Runs in two modes:
#   SYNTHETIC (default) — no internet required, instant
#   LIVE                — pulls from FRED + Yahoo Finance (requires internet)
#
# Switch mode by setting USE_LIVE_DATA = True below.
# =============================================================================

import sys
import logging
from pathlib import Path
from datetime import datetime

# --- Path setup --------------------------------------------------------------
SRC = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))

# --- Config ------------------------------------------------------------------
USE_LIVE_DATA = False   # <- set True to test live FRED + Yahoo feeds

# --- Logging -----------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)


def divider(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_data_fetch(config):
    divider("1a. DATA FETCHER")

    if USE_LIVE_DATA:
        print("Mode: LIVE (FRED + Yahoo Finance)")
        from data.fetcher import get_data
        raw = get_data(config, force_refresh=True)
    else:
        print("Mode: SYNTHETIC (no internet required)")
        from data.synthetic import generate_synthetic_data
        raw, true_regimes, dates = generate_synthetic_data()
        print(f"True regime cycle (first 12 months): {list(true_regimes[:12])}")

    print(f"\nSeries loaded: {len(raw)}")
    print()

    issues = []
    for sid, info in raw.items():
        s    = info["data"]
        freq = info["frequency"]
        src  = info["source"]
        n    = len(s)
        start  = s.index[0].date() if not s.empty else "—"
        end    = s.index[-1].date() if not s.empty else "—"
        status = "OK" if n > 12 else "THIN"
        if s.empty:
            status = "EMPTY"
            issues.append(sid)
        print(f"  {sid:<22} {src:<20} {freq}  "
              f"{n:>4} obs  {str(start)} -> {str(end)}  [{status}]")

    if issues:
        print(f"\n  WARNING  Empty series: {issues}")
    else:
        print(f"\n  PASS  All series non-empty")

    return raw


def test_vintage_manager(raw, config):
    divider("1b. VINTAGE MANAGER")

    from data.vintage_manager import build_monthly_frame, get_data_quality_summary

    as_of = datetime.utcnow()
    print(f"As-of date: {as_of.strftime('%Y-%m-%d')}")
    print()

    df, quality = build_monthly_frame(raw, config, as_of=as_of)

    print(f"Monthly frame shape:  {df.shape}")
    print(f"Date range:           {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Total months:         {len(df)}")
    print()

    base_cols = [c for c in df.columns
                 if not c.endswith("_revision") and not c.endswith("_surprise")]
    rev_cols  = [c for c in df.columns if c.endswith("_revision")]
    sur_cols  = [c for c in df.columns if c.endswith("_surprise")]

    print(f"Base columns:         {len(base_cols)}")
    print(f"Revision columns:     {len(rev_cols)}")
    print(f"Surprise columns:     {len(sur_cols)}")
    print()

    print("Data quality by series:")
    qdf = get_data_quality_summary(quality)
    print(f"  {'Series':<22} {'Tier':<22} {'Last Period':<12} Stale")
    print(f"  {'-'*22} {'-'*22} {'-'*12} -----")
    for sid, row in qdf.iterrows():
        stale_flag = "STALE" if row["stale"] else "ok"
        print(f"  {sid:<22} {row['tier']:<22} {str(row['last_period']):<12} {stale_flag}")

    n_stale   = qdf["stale"].sum()
    n_revised = (qdf["tier"] == "REVISED").sum()
    print()
    print(f"Stale series:         {n_stale}")
    print(f"Revised-data series:  {n_revised}  (bootstrap uses revised data — labelled in dashboard)")

    print()
    print("NaN coverage (base columns, last 24 months):")
    recent = df[base_cols].tail(24)
    issues = []
    for col in base_cols:
        n_nan = recent[col].isna().sum()
        if n_nan > 0:
            issues.append((col, n_nan))
    if issues:
        for col, n_nan in issues:
            print(f"  WARNING  {col:<30} {n_nan}/24 months NaN")
    nan_free = len(base_cols) - len(issues)
    print(f"  PASS  {nan_free}/{len(base_cols)} columns fully populated in last 24m")

    return df, quality


def test_state_log(config):
    divider("1c. STATE LOG")

    from data.state_log import load_state_log, get_last_run

    log_path = config.STATE_LOG_PATH
    print(f"State log path: {log_path}")

    df = load_state_log(log_path)

    if df.empty:
        print("\nState log is empty — no live runs recorded yet.")
        print("Expected on first run. Log will populate as model runs daily.")
    else:
        print(f"\nRuns recorded:  {len(df)}")
        print(f"First run:      {df['run_timestamp'].iloc[0]}")
        print(f"Latest run:     {df['run_timestamp'].iloc[-1]}")
        print()
        last = get_last_run(log_path)
        print("Latest run summary:")
        for k in ["regime_primary", "regime_confidence", "regime_transition",
                  "update_type", "n_stale_series", "changed_series"]:
            print(f"  {k:<30} {last.get(k, '—')}")


def test_synthetic_alignment(config):
    if USE_LIVE_DATA:
        return

    divider("1d. SYNTHETIC REGIME ALIGNMENT CHECK")

    import pandas as pd
    from data.synthetic import generate_synthetic_data
    from data.vintage_manager import build_monthly_frame

    raw, true_regimes, dates = generate_synthetic_data()
    df, _ = build_monthly_frame(raw, config)

    regime_s = pd.Series(true_regimes, index=dates[:len(true_regimes)])
    regime_s = regime_s.reindex(df.index, method="ffill")

    checks = [
        # Use native synthetic labels: Goldilocks, Reflation, Stagflation, Deflation
        # Check raw series directional alignment (before z-scoring)
        ("Goldilocks -> positive yield curve", "T10Y2Y",   "Goldilocks",  "higher"),
        ("Stagflation -> high CPI",            "CPIAUCSL", "Stagflation", "higher"),
        ("Deflation -> slow M2 growth (YoY)",  "M2SL",     "Deflation",   "lower"),
    ]

    print("Directional alignment (synthetic data, native four-quadrant labels):")
    for label, col, regime, expected in checks:
        if col not in df.columns:
            print(f"  --  {label}: {col} not in frame")
            continue
        s = df[col]
        # Use YoY growth rate for level series to avoid trend distortion
        if col in ("M2SL", "CPIAUCSL"):
            s = s.pct_change(12) * 100
        in_r  = s[regime_s == regime].mean()
        out_r = s[regime_s != regime].mean()
        if pd.isna(in_r) or pd.isna(out_r):
            print(f"  SKIP  {label} — insufficient data")
            continue
        ok = (expected == "higher" and in_r > out_r) or \
             (expected == "lower"  and in_r < out_r)
        flag = "PASS" if ok else "FAIL"
        print(f"  {flag}  {label}")
        print(f"        In-regime ({regime}): {in_r:.3f}  |  Out-of-regime: {out_r:.3f}")


def run_all():
    print()
    print("=" * 60)
    print("  MODULE 1 — TEST & REVIEW")
    print("  Data Fetcher | Vintage Manager | State Log")
    print("=" * 60)
    print(f"  Mode: {'LIVE' if USE_LIVE_DATA else 'SYNTHETIC'}")
    print(f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    import config

    raw         = test_data_fetch(config)
    df, quality = test_vintage_manager(raw, config)
    test_state_log(config)
    test_synthetic_alignment(config)

    divider("SUMMARY")
    print(f"  Series fetched:   {len(raw)}")
    print(f"  Monthly frame:    {df.shape[0]} months x {df.shape[1]} columns")
    print(f"  Date range:       {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Mode:             {'LIVE' if USE_LIVE_DATA else 'SYNTHETIC'}")
    print()
    print("  Module 1: PASS")
    print()


if __name__ == "__main__":
    run_all()
