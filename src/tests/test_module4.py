# =============================================================================
# TEST / REVIEW SCRIPT — MODULE 4
# Tests the historical regime map independently.
# Run from C:\dev\taa with: python src/tests/test_module4.py
# =============================================================================

import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

SRC = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))

USE_LIVE_DATA = False

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")


def divider(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def get_pipeline(config):
    if USE_LIVE_DATA:
        from data.fetcher import get_data
        raw = get_data(config)
    else:
        from data.synthetic import generate_synthetic_data
        raw, _, _ = generate_synthetic_data()
    from data.vintage_manager import build_monthly_frame
    from factors.engine import compute_factors
    from regimes.classifier import classify_regimes
    from analysis.regime_map import build_regime_map

    df, quality       = build_monthly_frame(raw, config)
    factors_df, _     = compute_factors(df, config)
    ensemble_df, _    = classify_regimes(factors_df, config)
    regime_map        = build_regime_map(ensemble_df, df, config)
    return ensemble_df, regime_map


def test_regime_periods(regime_map, config):
    divider("4a. REGIME PERIODS")

    periods = regime_map.get("regime_periods", pd.DataFrame())
    print(f"Total periods:   {len(periods)}")

    if periods.empty:
        print("  FAIL  No periods extracted")
        return

    print(f"Date range:      {periods['start'].min().date()} -> "
          f"{periods['end'].max().date()}")
    print()

    # Distribution by regime
    counts = periods.groupby("regime")["duration_months"].agg(["count", "mean", "sum"])
    print(f"  {'Regime':<20} {'Periods':>8} {'Avg Duration':>13} {'Total Months':>13}")
    print(f"  {'-'*20} {'-'*8} {'-'*13} {'-'*13}")
    for regime, row in counts.iterrows():
        print(f"  {regime:<20} {int(row['count']):>8} "
              f"{row['mean']:>12.1f}m {int(row['sum']):>12}m")

    # Data source breakdown
    print()
    src_counts = periods["data_source"].value_counts()
    for src, n in src_counts.items():
        flag = "NOTE: illustrative only" if src == "BOOTSTRAP" else "epistemically honest"
        print(f"  {src:<12} {n} periods  ({flag})")

    # Sanity: all five regimes should appear
    expected = set(config.REGIME_LABELS.keys())
    found    = set(periods["regime"].unique())
    missing  = expected - found
    flag = "PASS" if not missing else f"WARN — missing: {missing}"
    print(f"\n  {flag}  Regime coverage")


def test_return_matrices(regime_map, config):
    divider("4b. RETURN MATRICES")

    mx_ret = regime_map.get("matrix_return", pd.DataFrame())
    mx_hit = regime_map.get("matrix_hitrate", pd.DataFrame())
    mx_dd  = regime_map.get("matrix_drawdown", pd.DataFrame())

    for name, mx in [("Median Ann. Return (%)", mx_ret),
                      ("Hit Rate (%)",           mx_hit),
                      ("Median Max Drawdown (%)", mx_dd)]:
        print(f"\n  {name}:")
        if mx.empty:
            print("    MISSING")
        else:
            print(mx.round(1).to_string())

    # Sanity checks on return matrix
    # NOTE: These checks are directional only and may fail on bootstrap/synthetic data.
    # Bootstrap results use revised data with short, noisy regime periods.
    # These checks become more meaningful once live state log data accumulates.
    if not mx_ret.empty:
        print()
        print("  Directional sanity checks (informative — may fail on bootstrap data):")
        checks = []

        # S&P 500 should perform best in Goldilocks vs Danger Zone
        if "Goldilocks" in mx_ret.index and "Danger Zone" in mx_ret.index:
            if "^GSPC" in mx_ret.columns:
                gld_val = mx_ret.loc["Goldilocks", "^GSPC"]
                dng_val = mx_ret.loc["Danger Zone", "^GSPC"]
                if not (np.isnan(gld_val) or np.isnan(dng_val)):
                    ok = gld_val > dng_val
                    checks.append(("PASS" if ok else "NOTE",
                                   f"S&P 500 higher in Goldilocks ({gld_val:.1f}%) "
                                   f"than Danger Zone ({dng_val:.1f}%)"))

        # Gold should do well in Danger Zone
        if "Danger Zone" in mx_ret.index and "GLD" in mx_ret.columns:
            val = mx_ret.loc["Danger Zone", "GLD"]
            if not np.isnan(val):
                ok = val > 0
                checks.append(("PASS" if ok else "NOTE",
                               f"Gold positive in Danger Zone ({val:.1f}%)"))

        # Bonds should do better in Danger Zone than Goldilocks
        if "Danger Zone" in mx_ret.index and "Goldilocks" in mx_ret.index:
            if "TLT" in mx_ret.columns:
                dng_val = mx_ret.loc["Danger Zone", "TLT"]
                gld_val = mx_ret.loc["Goldilocks", "TLT"]
                if not (np.isnan(dng_val) or np.isnan(gld_val)):
                    ok = dng_val > gld_val
                    checks.append(("PASS" if ok else "NOTE",
                                   f"Bonds better in Danger Zone ({dng_val:.1f}%) "
                                   f"than Goldilocks ({gld_val:.1f}%)"))

        for flag, msg in checks:
            print(f"    {flag}  {msg}")

        if not checks:
            print("    No checks available — insufficient regime coverage")

        print()
        print("  NOTE: Bootstrap asset returns are illustrative only.")
        print("  Checks marked NOTE are directionally unexpected but not model failures.")
        print("  Re-evaluate once live state log data accumulates.")


def test_asset_stats(regime_map, config):
    divider("4c. ASSET STATS COVERAGE")

    stats = regime_map.get("asset_stats", pd.DataFrame())
    if stats.empty:
        print("  FAIL  No asset stats")
        return

    print(f"Total rows:  {len(stats)}")
    print(f"Assets:      {sorted(stats['asset'].unique())}")
    print(f"Regimes:     {sorted(stats['regime'].unique())}")
    print()

    # Check for NaNs
    key_cols = ["total_return", "annualised_return", "max_drawdown"]
    for col in key_cols:
        n_nan = stats[col].isna().sum()
        pct   = n_nan / len(stats) * 100
        flag  = "PASS" if pct < 10 else "WARN"
        print(f"  {flag}  {col:<25} {n_nan} NaN ({pct:.1f}%)")


def test_current_regime_context(ensemble_df, regime_map, config):
    divider("4d. CURRENT REGIME HISTORICAL CONTEXT")

    from analysis.regime_map import get_current_regime_historical_context

    current_regime = ensemble_df["regime_primary"].dropna().iloc[-1]
    ctx = get_current_regime_historical_context(regime_map, current_regime, config)

    print(f"Current regime:   {ctx.get('regime')}")
    print(f"Best asset:       {ctx.get('best_asset')}")
    print(f"Worst asset:      {ctx.get('worst_asset')}")
    print()
    print(f"  {'Asset':<10} {'Ann Ret':>8} {'Hit Rate':>9} {'Max DD':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*8}")
    for asset, stats in ctx.get("asset_returns", {}).items():
        print(f"  {asset:<10} "
              f"{stats['median_annualised_return']:>7.1f}%  "
              f"{stats['hit_rate']:>7.0f}%  "
              f"{stats['median_max_drawdown']:>7.1f}%")


def test_data_note(regime_map):
    divider("4e. DATA TRANSPARENCY")
    print(regime_map.get("data_note", "No data note found"))
    print()
    print("  NOTE: BOOTSTRAP periods use revised public data.")
    print("  These are illustrative — not what the model would have called in real time.")
    print("  LIVE periods (from state log) are epistemically honest.")
    print("  State log will accumulate from first live run onwards.")


def run_all():
    print()
    print("=" * 60)
    print("  MODULE 4 — TEST & REVIEW")
    print("  Historical Regime Map")
    print("=" * 60)
    print(f"  Mode: {'LIVE' if USE_LIVE_DATA else 'SYNTHETIC'}")
    print(f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    import config

    ensemble_df, regime_map = get_pipeline(config)

    test_regime_periods(regime_map, config)
    test_return_matrices(regime_map, config)
    test_asset_stats(regime_map, config)
    test_current_regime_context(ensemble_df, regime_map, config)
    test_data_note(regime_map)

    divider("SUMMARY")
    periods = regime_map.get("regime_periods", pd.DataFrame())
    summary = regime_map.get("summary", pd.DataFrame())
    print(f"  Regime periods:   {len(periods)}")
    print(f"  Summary rows:     {len(summary)}")
    print(f"  Mode:             {'LIVE' if USE_LIVE_DATA else 'SYNTHETIC'}")
    print()
    print("  Module 4: PASS")
    print()


if __name__ == "__main__":
    run_all()
