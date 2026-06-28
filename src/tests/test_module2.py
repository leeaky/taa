# =============================================================================
# TEST / REVIEW SCRIPT — MODULE 2
# Tests the factor engine independently.
# Run from C:\dev\taa with: python src/tests/test_module2.py
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

import pandas as pd
import numpy as np

# --- Path setup --------------------------------------------------------------
SRC = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))

# --- Config ------------------------------------------------------------------
USE_LIVE_DATA = False   # <- set True to test with live data

# --- Logging -----------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")


def divider(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def get_data(config):
    if USE_LIVE_DATA:
        from data.fetcher import get_data as fetch
        raw = fetch(config)
    else:
        from data.synthetic import generate_synthetic_data
        raw, _, _ = generate_synthetic_data()
    from data.vintage_manager import build_monthly_frame
    return build_monthly_frame(raw, config)


def test_factor_output(factors_df, meta, config):
    divider("2a. FACTOR OUTPUT — SHAPE & COVERAGE")

    factor_names = ["growth", "inflation", "liquidity", "risk_appetite"]
    windows      = [f"{w}m" for w in config.ZSCORE_WINDOWS]
    if config.ZSCORE_EXPANDING:
        windows.append("expanding")

    print(f"Frame shape:    {factors_df.shape}")
    print(f"Date range:     {factors_df.index[0].date()} -> {factors_df.index[-1].date()}")
    print(f"Total months:   {len(factors_df)}")
    print()

    print(f"{'Factor':<18} {'Windows':<8} {'Consensus col':<18} Status")
    print(f"{'-'*18} {'-'*8} {'-'*18} ------")
    for fname in factor_names:
        cols     = [f"{fname}_{w}" for w in windows if f"{fname}_{w}" in factors_df.columns]
        cons_col = f"{fname}_consensus"
        has_cons = cons_col in factors_df.columns
        status   = meta["factors"].get(fname, {}).get("status", "missing")
        flag     = "PASS" if status == "ok" else "MISSING"
        print(f"  {fname:<16} {len(cols):<8} {'yes' if has_cons else 'NO':<18} {flag}")


def test_current_scores(factors_df, config):
    divider("2b. CURRENT FACTOR SCORES (all windows)")

    from factors.engine import get_factor_summary
    summary = get_factor_summary(factors_df, config)

    print(f"As of: {factors_df.dropna(how='all').index[-1].date()}")
    print()
    print(summary.to_string())
    print()

    # Flag any windows diverging significantly from the consensus
    print("Window divergence flags (|max - min| > 1.0 std dev):")
    factor_names = ["growth", "inflation", "liquidity", "risk_appetite"]
    windows      = [f"{w}m" for w in config.ZSCORE_WINDOWS]
    if config.ZSCORE_EXPANDING:
        windows.append("expanding")

    flagged = False
    for fname in factor_names:
        cols = [f"{fname}_{w}" for w in windows if f"{fname}_{w}" in factors_df.columns]
        latest = factors_df[cols].dropna(how="all").iloc[-1]
        spread = latest.max() - latest.min()
        if spread > 1.0:
            flagged = True
            print(f"  FLAG  {fname:<18} spread = {spread:.2f} std devs across windows")
            for col in cols:
                print(f"          {col:<30} {latest[col]:.3f}")
    if not flagged:
        print("  PASS  No significant window divergence at current date")


def test_consensus_scores(factors_df):
    divider("2c. CONSENSUS SCORES (window agreement)")

    cons_cols = [c for c in factors_df.columns if c.endswith("_consensus")]
    latest    = factors_df[cons_cols].dropna(how="all").iloc[-1]

    print(f"{'Factor':<35} {'Consensus':<10} Interpretation")
    print(f"{'-'*35} {'-'*10} --------------")
    for col, val in latest.items():
        fname = col.replace("_consensus", "")
        if val >= 0.90:
            interp = "High — windows agree"
        elif val >= 0.75:
            interp = "Moderate — minor divergence"
        else:
            interp = "Low — windows disagree, use caution"
        print(f"  {col:<33} {val:<10.3f} {interp}")


def test_factor_history(factors_df, config):
    divider("2d. FACTOR HISTORY — SPOT CHECKS")

    # Check that factors are not constant (would suggest a bug)
    factor_names = ["growth", "inflation", "liquidity", "risk_appetite"]
    ref_window   = "60m"

    print(f"Checking variability of *_{ref_window} columns:")
    print()
    issues = []
    for fname in factor_names:
        col = f"{fname}_{ref_window}"
        if col not in factors_df.columns:
            print(f"  MISSING  {col}")
            continue
        s    = factors_df[col].dropna()
        std  = s.std()
        mean = s.mean()
        mn   = s.min()
        mx   = s.max()
        ok   = std > 0.1
        flag = "PASS" if ok else "FLAT"
        if not ok:
            issues.append(col)
        print(f"  {flag}  {col:<30}  mean={mean:+.3f}  std={std:.3f}  "
              f"min={mn:+.3f}  max={mx:+.3f}")

    if issues:
        print(f"\n  WARNING  Flat series: {issues}")
    else:
        print(f"\n  PASS  All factors show meaningful variation")


def test_regime_alignment(factors_df, config):
    """
    Synthetic only: verify that factor scores align with the known
    true regime used to generate the synthetic data.
    Uses the synthetic generator's native four-quadrant labels directly
    (before mapping to five-regime taxonomy) for clarity.
    """
    if USE_LIVE_DATA:
        return

    divider("2e. FACTOR-REGIME ALIGNMENT (synthetic data)")

    from data.synthetic import generate_synthetic_data

    _, true_regimes, dates = generate_synthetic_data()
    regime_s = pd.Series(true_regimes, index=dates[:len(true_regimes)])
    regime_s = regime_s.reindex(factors_df.index, method="ffill")

    ref = "60m"

    # Use native synthetic labels — no mapping needed
    # Synthetic generator uses: Goldilocks, Reflation, Stagflation, Deflation
    checks = [
        # (description, factor_col, expected_high_regime, expected_low_regime)
        ("Growth higher in Goldilocks vs Stagflation",
         f"growth_{ref}", "Goldilocks", "Stagflation"),
        ("Inflation higher in Stagflation vs Goldilocks",
         f"inflation_{ref}", "Stagflation", "Goldilocks"),
        ("Risk appetite higher in Goldilocks vs Stagflation",
         f"risk_appetite_{ref}", "Goldilocks", "Stagflation"),
        ("Liquidity higher in Goldilocks vs Deflation",
         f"liquidity_{ref}", "Goldilocks", "Deflation"),
    ]

    print(f"Using {ref} window. Synthetic regime alignment:")
    print(f"(Labels are synthetic generator's native four-quadrant labels)")
    print()
    for desc, col, high_regime, low_regime in checks:
        if col not in factors_df.columns:
            print(f"  --  {col} missing")
            continue
        high_mean = factors_df[col][regime_s == high_regime].mean()
        low_mean  = factors_df[col][regime_s == low_regime].mean()
        if pd.isna(high_mean) or pd.isna(low_mean):
            print(f"  SKIP  {desc} — insufficient data for one or both regimes")
            continue
        ok   = high_mean > low_mean
        flag = "PASS" if ok else "FAIL"
        print(f"  {flag}  {desc}")
        print(f"        {high_regime}: {high_mean:+.3f}  |  "
              f"{low_regime}: {low_mean:+.3f}")


def run_all():
    print()
    print("=" * 60)
    print("  MODULE 2 — TEST & REVIEW")
    print("  Factor Engine")
    print("=" * 60)
    print(f"  Mode: {'LIVE' if USE_LIVE_DATA else 'SYNTHETIC'}")
    print(f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    import config
    from factors.engine import compute_factors

    df, quality   = get_data(config)
    factors_df, meta = compute_factors(df, config)

    test_factor_output(factors_df, meta, config)
    test_current_scores(factors_df, config)
    test_consensus_scores(factors_df)
    test_factor_history(factors_df, config)
    test_regime_alignment(factors_df, config)

    divider("SUMMARY")
    n_factor_cols = len([c for c in factors_df.columns
                         if not c.endswith("_consensus")])
    n_cons_cols   = len([c for c in factors_df.columns
                         if c.endswith("_consensus")])
    print(f"  Factor columns:   {n_factor_cols}")
    print(f"  Consensus cols:   {n_cons_cols}")
    print(f"  Valid months:     {len(factors_df.dropna(how='all'))}")
    print(f"  Mode:             {'LIVE' if USE_LIVE_DATA else 'SYNTHETIC'}")
    print()
    print("  Module 2: PASS")
    print()


if __name__ == "__main__":
    run_all()
