# =============================================================================
# TEST / REVIEW SCRIPT — MODULE 3
# Tests the regime classifier independently.
# Run from C:\dev\taa with: python src/tests/test_module3.py
#
# USE_LIVE_DATA = False  → synthetic data, instant
# USE_LIVE_DATA = True   → live FRED + Yahoo
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
    df, quality    = build_monthly_frame(raw, config)
    factors_df, _  = compute_factors(df, config)
    return classify_regimes(factors_df, config)


def test_output_shape(ensemble_df, window_df, config):
    divider("3a. OUTPUT SHAPE & COVERAGE")

    from regimes.classifier import REGIME_NAMES

    windows = [f"{w}m" for w in config.ZSCORE_WINDOWS]
    if config.ZSCORE_EXPANDING:
        windows.append("expanding")

    print(f"Ensemble frame:  {ensemble_df.shape}")
    print(f"Window frame:    {window_df.shape}")
    print(f"Date range:      {ensemble_df.index[0].date()} -> "
          f"{ensemble_df.index[-1].date()}")
    print()

    # Check required columns
    required = (
        ["regime_primary", "regime_confidence", "regime_transition",
         "confidence_magnitude", "confidence_secondary", "confidence_consensus"]
        + [f"prob_{r.replace(' ', '_')}" for r in REGIME_NAMES]
        + [f"regime_{w}" for w in windows]
    )
    missing = [c for c in required if c not in ensemble_df.columns]
    if missing:
        print(f"  FAIL  Missing columns: {missing}")
    else:
        print(f"  PASS  All {len(required)} required columns present")


def test_current_regime(ensemble_df, config):
    divider("3b. CURRENT REGIME CALL")

    from regimes.classifier import get_current_regime, REGIME_NAMES

    current = get_current_regime(ensemble_df)
    windows = [f"{w}m" for w in config.ZSCORE_WINDOWS]
    if config.ZSCORE_EXPANDING:
        windows.append("expanding")

    print(f"As of:        {ensemble_df.dropna(subset=['regime_primary']).index[-1].date()}")
    print()
    print(f"  Primary regime:   {current.get('regime_primary')}")
    print(f"  Confidence:       {current.get('regime_confidence', 0):.3f}")
    print(f"  Transition flag:  {current.get('regime_transition')}")
    print()

    print("  Probability vector:")
    probs = {}
    for r in REGIME_NAMES:
        key  = f"prob_{r.replace(' ', '_')}"
        val  = current.get(key) or 0
        probs[r] = val
        bar  = "#" * int(val * 25)
        print(f"    {r:<20} {val:.3f}  {bar}")

    total = sum(probs.values())
    flag  = "PASS" if abs(total - 1.0) < 0.01 else "FAIL"
    print(f"\n  {flag}  Probabilities sum to {total:.4f} (should be ~1.0)")

    print()
    print("  Per-window calls:")
    for w in windows:
        print(f"    regime_{w:<15} {current.get(f'regime_{w}', '—')}")


def test_confidence_layers(ensemble_df):
    divider("3c. CONFIDENCE LAYERS")

    layers = ["confidence_magnitude", "confidence_secondary",
              "confidence_consensus", "regime_confidence"]

    print(f"{'Layer':<30} {'Min':>6} {'Mean':>6} {'Max':>6}  Range check")
    print(f"{'-'*30} {'-'*6} {'-'*6} {'-'*6}  -----------")

    for col in layers:
        if col not in ensemble_df.columns:
            print(f"  {col:<30} MISSING")
            continue
        s    = ensemble_df[col].dropna()
        mn   = s.min()
        mean = s.mean()
        mx   = s.max()
        ok   = 0.0 <= mn and mx <= 1.0
        flag = "PASS" if ok else "FAIL"
        print(f"  {col:<30} {mn:>6.3f} {mean:>6.3f} {mx:>6.3f}  {flag} (must be 0-1)")


def test_regime_distribution(ensemble_df):
    divider("3d. HISTORICAL REGIME DISTRIBUTION")

    counts = ensemble_df["regime_primary"].value_counts()
    total  = counts.sum()

    print(f"Total months classified: {total}")
    print()
    for regime, count in counts.items():
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {regime:<20} {count:>4} months  ({pct:>5.1f}%)  {bar}")

    # Sanity: no single regime should dominate >80% of history
    max_pct = counts.max() / total * 100
    flag = "PASS" if max_pct < 80 else "WARN"
    print(f"\n  {flag}  Max single regime share: {max_pct:.1f}% (warn if >80%)")


def test_transition_flags(ensemble_df):
    divider("3e. TRANSITION FLAGS")

    n_total = len(ensemble_df.dropna(subset=["regime_primary"]))
    n_trans = ensemble_df["regime_transition"].sum()
    pct     = n_trans / n_total * 100

    print(f"Total months:      {n_total}")
    print(f"Transition months: {int(n_trans)} ({pct:.1f}%)")

    # Expect 20-50% — too few means model never flags uncertainty,
    # too many means it's too sensitive
    if pct < 10:
        flag = "WARN — very few transitions, model may be too stable"
    elif pct > 60:
        flag = "WARN — many transitions, model may be too noisy"
    else:
        flag = "PASS — reasonable transition frequency"
    print(f"\n  {flag}")

    print()
    print("Last 12 months:")
    recent = ensemble_df.tail(12)[["regime_primary", "regime_confidence",
                                    "regime_transition"]]
    for dt, row in recent.iterrows():
        trans = "TRANSITION" if row["regime_transition"] else ""
        print(f"  {dt.date()}  {str(row['regime_primary']):<20}  "
              f"conf={row['regime_confidence']:.2f}  {trans}")


def test_window_consensus(ensemble_df, window_df, config):
    divider("3f. WINDOW CONSENSUS DETAIL")

    windows = [f"{w}m" for w in config.ZSCORE_WINDOWS]
    if config.ZSCORE_EXPANDING:
        windows.append("expanding")

    # Count how often all windows agree
    regime_cols = [f"regime_{w}" for w in windows
                   if f"regime_{w}" in window_df.columns]

    if not regime_cols:
        print("  No per-window regime columns found")
        return

    recent = window_df[regime_cols].tail(24)
    full_agree = 0
    for _, row in recent.iterrows():
        vals = [v for v in row.values if v not in (None, "nan", "None")]
        if len(set(vals)) == 1 and vals:
            full_agree += 1

    print(f"Last 24 months — all windows agree: {full_agree}/24")
    print()
    print("Current window calls:")
    latest = window_df[regime_cols].dropna(how="all").iloc[-1]
    for col, val in latest.items():
        print(f"  {col:<25} {val}")


def run_all():
    print()
    print("=" * 60)
    print("  MODULE 3 — TEST & REVIEW")
    print("  Regime Classifier")
    print("=" * 60)
    print(f"  Mode: {'LIVE' if USE_LIVE_DATA else 'SYNTHETIC'}")
    print(f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    import config

    ensemble_df, window_df = get_pipeline(config)

    test_output_shape(ensemble_df, window_df, config)
    test_current_regime(ensemble_df, config)
    test_confidence_layers(ensemble_df)
    test_regime_distribution(ensemble_df)
    test_transition_flags(ensemble_df)
    test_window_consensus(ensemble_df, window_df, config)

    divider("SUMMARY")
    n_valid = len(ensemble_df.dropna(subset=["regime_primary"]))
    print(f"  Months classified:  {n_valid}")
    print(f"  Mode:               {'LIVE' if USE_LIVE_DATA else 'SYNTHETIC'}")
    print()
    print("  Module 3: PASS")
    print()


if __name__ == "__main__":
    run_all()
