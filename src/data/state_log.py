# =============================================================================
# MODULE 1c: STATE LOG
# Append-only record of what the model output at every run.
# This is the live epistemic record — what the model actually knew and said.
# Never overwrites. Never back-adjusts.
# =============================================================================

import logging
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

COLUMNS = [
    "run_timestamp",       # UTC datetime of this model run
    "as_of_date",          # latest data date used
    "regime_primary",      # majority vote regime label
    "regime_confidence",   # 0-1 combined confidence
    "confidence_magnitude",
    "confidence_secondary",
    "confidence_consensus",
    "regime_transition",   # boolean — transition flagged
    # Factor scores — primary window (60m as default display)
    "growth_36m",  "growth_60m",  "growth_120m",  "growth_240m",  "growth_expanding",
    "inflation_36m","inflation_60m","inflation_120m","inflation_240m","inflation_expanding",
    "liquidity_36m","liquidity_60m","liquidity_120m","liquidity_240m","liquidity_expanding",
    "risk_appetite_36m","risk_appetite_60m","risk_appetite_120m","risk_appetite_240m","risk_appetite_expanding",
    # Regime variants
    "regime_36m",  "regime_60m",  "regime_120m",  "regime_240m",  "regime_expanding",
    # Data quality
    "n_stale_series",
    "n_revised_series",
    "update_type",         # FULL_UPDATE | NO_CHANGE | DATA_PENDING
    "changed_series",      # comma-separated list of series that changed this run
]


def append_run(
    path: Path,
    run_timestamp: datetime,
    as_of_date,
    regime_output: dict,
    factor_scores: dict,
    quality_summary: dict,
    changed_series: list,
    update_type: str = "FULL_UPDATE",
):
    """
    Append one row to the state log CSV.
    Creates the file with headers if it doesn't exist.

    Parameters
    ----------
    path            : path to state_log.csv
    run_timestamp   : datetime of this run
    as_of_date      : latest date for which data exists
    regime_output   : dict from classifier (regime_primary, confidence, etc.)
    factor_scores   : dict of {factor_window: score} e.g. {'growth_60m': 0.42}
    quality_summary : dict from vintage_manager
    changed_series  : list of series IDs that changed since last run
    update_type     : FULL_UPDATE | NO_CHANGE | DATA_PENDING
    """
    path = Path(path)
    write_header = not path.exists()

    n_stale   = sum(1 for v in quality_summary.values() if v.get("stale"))
    n_revised = sum(1 for v in quality_summary.values() if v.get("tier") == "REVISED")

    row = {
        "run_timestamp":        run_timestamp.isoformat(),
        "as_of_date":           str(as_of_date),
        "regime_primary":       regime_output.get("regime_primary", ""),
        "regime_confidence":    round(regime_output.get("regime_confidence", 0), 4),
        "confidence_magnitude": round(regime_output.get("confidence_magnitude", 0), 4),
        "confidence_secondary": round(regime_output.get("confidence_secondary", 0), 4),
        "confidence_consensus": round(regime_output.get("confidence_consensus", 0), 4),
        "regime_transition":    regime_output.get("regime_transition", False),
        "n_stale_series":       n_stale,
        "n_revised_series":     n_revised,
        "update_type":          update_type,
        "changed_series":       ",".join(changed_series) if changed_series else "",
    }

    # Factor scores
    for key, val in factor_scores.items():
        if key in COLUMNS:
            row[key] = round(float(val), 4) if pd.notna(val) else ""

    # Regime variants
    for key, val in regime_output.items():
        if key.startswith("regime_") and key in COLUMNS:
            row[key] = val

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"State log updated: {update_type} at {run_timestamp.isoformat()[:19]}")


def load_state_log(path: Path) -> pd.DataFrame:
    """
    Load the full state log. Returns empty DataFrame if no log exists yet.
    """
    path = Path(path)
    if not path.exists():
        logger.info("No state log yet — starting fresh.")
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path, parse_dates=["run_timestamp", "as_of_date"])
        df = df.sort_values("run_timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Could not load state log: {e}")
        return pd.DataFrame(columns=COLUMNS)


def get_last_run(path: Path) -> dict | None:
    """Return the most recent row of the state log as a dict."""
    df = load_state_log(path)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def detect_changes(current_raw: dict, last_run: dict | None) -> tuple[list, str]:
    """
    Compare current fetch to last run to detect what changed.

    Returns
    -------
    changed_series : list of series IDs with new data
    update_type    : FULL_UPDATE | NO_CHANGE | DATA_PENDING
    """
    if last_run is None:
        return list(current_raw.keys()), "FULL_UPDATE"

    changed = []
    for sid, info in current_raw.items():
        s = info["data"]
        if s.empty:
            continue
        last_date = s.index[-1]
        log_key   = f"last_date_{sid}"
        prev_date = last_run.get(log_key)
        if prev_date is None or str(last_date.date()) != str(prev_date):
            changed.append(sid)

    if not changed:
        return [], "NO_CHANGE"

    return changed, "FULL_UPDATE"
