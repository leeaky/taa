# =============================================================================
# DAILY RUNNER
# Runs the full TAA model pipeline once and appends one row to the state log.
# Designed to be scheduled via Windows Task Scheduler to run daily.
#
# Schedule: once per day, ideally after major data releases have landed
# (e.g. 8:00 AM ET covers most overnight FRED updates)
#
# Run manually from C:\dev\taa with:
#   python src/runner.py
#
# Schedule via Windows Task Scheduler:
#   Program:   python
#   Arguments: C:\dev\taa\src\runner.py
#   Start in:  C:\dev\taa
# =============================================================================

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# --- Path setup --------------------------------------------------------------
SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))

# --- Logging — writes to logs/runner.log as well as console ------------------
LOG_DIR = SRC / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "runner.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run():
    run_start = datetime.utcnow()
    logger.info("=" * 60)
    logger.info(f"TAA Daily Runner — {run_start.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    # --- Import after path setup so src/ modules resolve ---------------------
    import config_overrides as config
    from config_overrides import get_effective_config, load_overrides
    from data.fetcher import get_data
    from data.vintage_manager import build_monthly_frame
    from data.state_log import (
        load_state_log, append_run, get_last_run, detect_changes
    )
    from factors.engine import compute_factors, get_factor_summary
    from regimes.classifier import classify_regimes, get_current_regime, REGIME_NAMES

    # --- Build effective config (base + any overrides) -----------------------
    overrides = load_overrides()
    import types
    effective = get_effective_config()
    effective.update(overrides)
    cfg = types.SimpleNamespace(**effective)

    # --- Fetch data ----------------------------------------------------------
    logger.info("Fetching data...")
    try:
        raw = get_data(cfg, force_refresh=True)
        logger.info(f"Fetched {len(raw)} series")
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return False

    # --- Detect what changed since last run ----------------------------------
    last_run = get_last_run(cfg.STATE_LOG_PATH)
    changed_series, update_type = detect_changes(raw, last_run)

    if update_type == "NO_CHANGE":
        logger.info("No new data since last run — appending NO_CHANGE entry")
        append_run(
            path=cfg.STATE_LOG_PATH,
            run_timestamp=run_start,
            as_of_date=run_start.date(),
            regime_output={},
            factor_scores={},
            quality_summary={},
            changed_series=[],
            update_type="NO_CHANGE",
        )
        logger.info("Done.")
        return True

    logger.info(f"Changed series: {changed_series}")

    # --- Build monthly frame -------------------------------------------------
    logger.info("Building monthly frame...")
    try:
        df, quality = build_monthly_frame(raw, cfg)
        logger.info(f"Monthly frame: {df.shape[0]} months × {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Vintage manager failed: {e}")
        return False

    # --- Factor engine -------------------------------------------------------
    logger.info("Computing factors...")
    try:
        factors_df, meta = compute_factors(df, cfg)
        valid_rows = len(factors_df.dropna(how="all"))
        logger.info(f"Factor engine: {valid_rows} valid months")

        # Log current factor readings
        summary = get_factor_summary(factors_df, cfg)
        ref = "60m"
        for fname in ["growth", "inflation", "liquidity", "risk_appetite"]:
            col = f"{fname}_{ref}"
            if col in factors_df.columns:
                val = factors_df[col].dropna()
                if not val.empty:
                    logger.info(f"  {fname:<16} {val.iloc[-1]:+.3f}  "
                                f"(as of {val.index[-1].date()})")
    except Exception as e:
        logger.error(f"Factor engine failed: {e}")
        return False

    # --- Regime classifier ---------------------------------------------------
    logger.info("Classifying regimes...")
    try:
        ensemble_df, window_df = classify_regimes(factors_df, cfg)
        current = get_current_regime(ensemble_df)

        regime    = current.get("regime_primary", "—")
        conf      = current.get("regime_confidence", 0)
        trans     = current.get("regime_transition", False)
        logger.info(f"Regime: {regime}  confidence={conf:.2f}  transition={trans}")

        # Log probability vector
        for r in REGIME_NAMES:
            key = f"prob_{r.replace(' ', '_')}"
            val = current.get(key, 0) or 0
            logger.info(f"  {r:<20} {val:.1%}")
    except Exception as e:
        logger.error(f"Classifier failed: {e}")
        return False

    # --- Build factor scores dict for state log ------------------------------
    windows = [f"{w}m" for w in cfg.ZSCORE_WINDOWS]
    if cfg.ZSCORE_EXPANDING:
        windows.append("expanding")

    factor_scores = {}
    for fname in ["growth", "inflation", "liquidity", "risk_appetite"]:
        for w in windows:
            col = f"{fname}_{w}"
            if col in factors_df.columns:
                vals = factors_df[col].dropna()
                if not vals.empty:
                    factor_scores[col] = vals.iloc[-1]

    # --- Determine as_of date ------------------------------------------------
    as_of_date = factors_df.dropna(how="all").index[-1].date() \
        if not factors_df.dropna(how="all").empty else run_start.date()

    # --- Append to state log -------------------------------------------------
    logger.info(f"Appending to state log: {cfg.STATE_LOG_PATH}")
    try:
        append_run(
            path=cfg.STATE_LOG_PATH,
            run_timestamp=run_start,
            as_of_date=as_of_date,
            regime_output=current,
            factor_scores=factor_scores,
            quality_summary=quality,
            changed_series=changed_series,
            update_type=update_type,
        )
        logger.info("State log updated successfully")
    except Exception as e:
        logger.error(f"State log write failed: {e}")
        return False

    # --- Save last run metadata ----------------------------------------------
    last_run_meta = {
        "run_timestamp":  run_start.isoformat(),
        "as_of_date":     str(as_of_date),
        "regime_primary": regime,
        "confidence":     conf,
        "update_type":    update_type,
        "changed_series": changed_series,
    }
    # Add last-known dates per series for change detection
    for sid, info in raw.items():
        s = info["data"]
        if not s.empty:
            last_run_meta[f"last_date_{sid}"] = str(s.index[-1].date())

    with open(cfg.LAST_RUN_PATH, "w") as f:
        json.dump(last_run_meta, f, indent=2)

    # --- Summary -------------------------------------------------------------
    elapsed = (datetime.utcnow() - run_start).total_seconds()
    logger.info("-" * 60)
    logger.info(f"Run complete in {elapsed:.1f}s")
    logger.info(f"Regime:     {regime}")
    logger.info(f"Confidence: {conf:.0%}")
    logger.info(f"As of:      {as_of_date}")
    logger.info(f"Updated:    {len(changed_series)} series")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
