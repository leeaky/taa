# =============================================================================
# MODULE 3: REGIME CLASSIFIER
# Maps factor scores to a probability vector across five regimes.
#
# Fullerton's five regimes (confirmed across FIV Q1 2023 - Q2 2026):
#   Recovery / Early Cycle  — growth recovering, inflation contained
#   Goldilocks              — above-trend growth, below-trend inflation
#   Late Cycle              — decelerating growth, rising inflation
#   Danger Zone             — falling growth, elevated stress
#   Sentiment Driven        — macro ambiguous, risk appetite dominant
#
# Output per date:
#   - Probability vector across five regimes (sums to 1.0)
#   - Primary regime label (highest probability)
#   - Three-layer confidence score (magnitude, secondary, consensus)
#   - Transition flag (when consensus < threshold or recent label change)
#   - All five window variants (smoothed + raw)
# =============================================================================

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import softmax

logger = logging.getLogger(__name__)

REGIME_NAMES = [
    "Recovery",
    "Goldilocks",
    "Late Cycle",
    "Danger Zone",
    "Sentiment Driven",
]


# =============================================================================
# REGIME SCORING — PRIMARY CLASSIFICATION
# Growth and Inflation are the primary axes.
# Each regime has an expected (growth, inflation) signature.
# Score = distance from each regime's ideal point in factor space.
# =============================================================================

def _get_ideals(config) -> dict:
    """Primary (growth, inflation) ideal points — from config."""
    return config.REGIME_IDEALS


def _get_secondary(config) -> dict:
    """Secondary (liquidity, risk_appetite) ideal points — from config."""
    return config.REGIME_SECONDARY_IDEALS


def _distance_scores(growth: float, inflation: float, config) -> dict:
    """
    Compute inverse-distance score for each regime in (growth, inflation) space.
    Uses Euclidean distance with a softmax to convert to probabilities.
    """
    scores = {}
    for regime, (g_ideal, i_ideal) in _get_ideals(config).items():
        dist = np.sqrt((growth - g_ideal) ** 2 + (inflation - i_ideal) ** 2)
        scores[regime] = -dist
    return scores


def _sentiment_boost(risk_appetite: float, liquidity: float, config) -> float:
    """
    Boost for Sentiment Driven regime when risk appetite is elevated
    and macro factors are ambiguous. Parameters from config.
    """
    ra_signal = max(0, risk_appetite - config.SENTIMENT_RA_THRESHOLD)
    return ra_signal * config.SENTIMENT_BOOST_WEIGHT


def classify_single(
    growth: float,
    inflation: float,
    liquidity: float,
    risk_appetite: float,
    config=None,
) -> dict:
    """
    Classify a single observation into regime probabilities.
    """
    if any(np.isnan(x) for x in [growth, inflation, liquidity, risk_appetite]):
        return {
            "probabilities": {r: np.nan for r in REGIME_NAMES},
            "primary":       None,
            "raw_scores":    {},
        }

    scores = _distance_scores(growth, inflation, config)

    macro_ambiguity = np.exp(-0.5 * (growth**2 + inflation**2))
    scores["Sentiment Driven"] += (
        _sentiment_boost(risk_appetite, liquidity, config) * macro_ambiguity
    )

    names  = list(scores.keys())
    values = np.array([scores[n] for n in names])
    probs  = softmax(values / config.SOFTMAX_TEMPERATURE)

    prob_dict = {n: float(p) for n, p in zip(names, probs)}
    primary   = max(prob_dict, key=prob_dict.get)

    return {
        "probabilities": prob_dict,
        "primary":       primary,
        "raw_scores":    scores,
    }


# =============================================================================
# CONFIDENCE SCORING — THREE LAYERS
# =============================================================================

def confidence_magnitude(
    growth: float,
    inflation: float,
    primary: str,
    config=None,
) -> float:
    """Layer 1: distance of factors from origin, scaled by ideal distance."""
    if primary is None or any(np.isnan(x) for x in [growth, inflation]):
        return 0.0

    ideals = _get_ideals(config)
    g_ideal, i_ideal = ideals.get(primary, (0.0, 0.0))
    ideal_dist  = np.sqrt(g_ideal**2 + i_ideal**2)
    actual_dist = np.sqrt(growth**2 + inflation**2)

    if ideal_dist == 0:
        return float(np.clip(actual_dist / 1.0, 0, 1))

    return float(np.clip(actual_dist / (ideal_dist + 0.5), 0, 1))


def confidence_secondary(
    liquidity: float,
    risk_appetite: float,
    primary: str,
    config=None,
) -> float:
    """Layer 2: do secondary factors support the primary regime?"""
    if primary is None or any(np.isnan(x) for x in [liquidity, risk_appetite]):
        return 0.5

    secondary = _get_secondary(config)
    l_exp, ra_exp = secondary.get(primary, (0.0, 0.0))

    l_align  = 1 - np.clip(abs(liquidity - l_exp)  / 2.0, 0, 1)
    ra_align = 1 - np.clip(abs(risk_appetite - ra_exp) / 2.0, 0, 1)

    return float((l_align + ra_align) / 2)


def confidence_consensus(window_primaries: list) -> float:
    """
    Layer 3: What fraction of window variants agree on the primary regime?
    5/5 = 1.0, 3/5 = 0.6, etc.
    """
    valid = [r for r in window_primaries if r is not None]
    if not valid:
        return 0.0
    most_common = max(set(valid), key=valid.count)
    return sum(1 for r in valid if r == most_common) / len(valid)


def combined_confidence(mag: float, sec: float, cons: float) -> float:
    """Equal-weighted combination of three confidence layers."""
    return float((mag + sec + cons) / 3)


# =============================================================================
# SMOOTHING — ROLLING MODE
# Suppress whipsawing between regimes on noisy monthly data.
# =============================================================================

def rolling_mode(s: pd.Series, window: int) -> pd.Series:
    """
    Rolling mode of a string series — pandas 2.x compatible.
    At each point, returns the most frequent label in the prior `window` months.
    """
    values = list(s)
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = [v for v in values[start:i + 1]
                       if v is not None and str(v) not in ("nan", "None")]
        if not window_vals:
            result.append(None)
        else:
            result.append(max(set(window_vals), key=window_vals.count))
    return pd.Series(result, index=s.index)


# =============================================================================
# MAIN CLASSIFICATION PIPELINE — PER WINDOW
# =============================================================================

def classify_window(
    factors_df: pd.DataFrame,
    window_label: str,
    config,
) -> pd.DataFrame:
    """
    Classify all dates for a single window variant.

    Parameters
    ----------
    factors_df   : output of factor engine
    window_label : e.g. '60m', 'expanding'
    config       : config module

    Returns
    -------
    DataFrame with columns:
        regime_{window}              — smoothed primary label
        raw_regime_{window}          — unsmoothed primary label
        prob_{regime}_{window}       — probability for each of five regimes
        confidence_magnitude_{window}
        confidence_secondary_{window}
    """
    g_col  = f"growth_{window_label}"
    i_col  = f"inflation_{window_label}"
    l_col  = f"liquidity_{window_label}"
    ra_col = f"risk_appetite_{window_label}"

    missing = [c for c in [g_col, i_col, l_col, ra_col]
               if c not in factors_df.columns]
    if missing:
        logger.warning(f"Window {window_label}: missing columns {missing}")
        return pd.DataFrame(index=factors_df.index)

    rows = []
    for dt in factors_df.index:
        g  = factors_df.loc[dt, g_col]
        i  = factors_df.loc[dt, i_col]
        l  = factors_df.loc[dt, l_col]
        ra = factors_df.loc[dt, ra_col]

        result = classify_single(g, i, l, ra, config)
        probs  = result["probabilities"]
        prim   = result["primary"]

        mag = confidence_magnitude(g, i, prim, config)
        sec = confidence_secondary(l, ra, prim, config)

        row = {f"raw_regime_{window_label}": prim,
               f"confidence_magnitude_{window_label}": mag,
               f"confidence_secondary_{window_label}": sec}
        for rname in REGIME_NAMES:
            row[f"prob_{rname.replace(' ', '_')}_{window_label}"] = probs.get(rname, np.nan)

        rows.append(row)

    df_out = pd.DataFrame(rows, index=factors_df.index)

    # Smooth the primary label
    raw_col = f"raw_regime_{window_label}"
    df_out[f"regime_{window_label}"] = rolling_mode(
        df_out[raw_col], config.SMOOTHING_WINDOW
    )

    return df_out


# =============================================================================
# ENSEMBLE — COMBINE ALL WINDOWS
# =============================================================================

def classify_all_windows(
    factors_df: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Run classification across all configured windows.
    Returns wide DataFrame with all window variants.
    """
    windows = [f"{w}m" for w in config.ZSCORE_WINDOWS]
    if config.ZSCORE_EXPANDING:
        windows.append("expanding")

    parts = []
    for w in windows:
        logger.info(f"Classifying window: {w}")
        df_w = classify_window(factors_df, w, config)
        if not df_w.empty:
            parts.append(df_w)

    if not parts:
        logger.error("Classifier: no window output produced")
        return pd.DataFrame(index=factors_df.index)

    return pd.concat(parts, axis=1)


# =============================================================================
# ENSEMBLE OUTPUT — PRIMARY CALL + PROBABILITIES + CONFIDENCE
# =============================================================================

def compute_ensemble(
    window_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Combine all window variants into a single ensemble output per date.

    Ensemble probability = average of per-window probabilities.
    Primary = highest ensemble probability regime.
    Confidence = three-layer combined score.
    Transition flag = consensus < threshold OR label changed in last 2 months.

    Returns
    -------
    DataFrame with columns:
        regime_primary
        regime_confidence
        confidence_magnitude
        confidence_secondary
        confidence_consensus
        regime_transition
        prob_{regime}             — ensemble probability per regime
        regime_{window}           — smoothed label per window (for display)
    """
    windows = [f"{w}m" for w in config.ZSCORE_WINDOWS]
    if config.ZSCORE_EXPANDING:
        windows.append("expanding")

def compute_ensemble(
    window_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Combine all window variants into a single ensemble output per date.
    Fully vectorised — no row-by-row loop.
    """
    windows = [f"{w}m" for w in config.ZSCORE_WINDOWS]
    if config.ZSCORE_EXPANDING:
        windows.append("expanding")

    out = pd.DataFrame(index=window_df.index)

    # --- Ensemble probabilities (weighted mean across windows) ---------------
    window_weights = getattr(config, "ZSCORE_WINDOW_WEIGHTS", {})
    weights = {w: window_weights.get(w, 1.0) for w in windows}

    for rname in REGIME_NAMES:
        safe = rname.replace(" ", "_")
        weighted_sum   = None
        active_weight  = 0.0
        for w in windows:
            col = f"prob_{safe}_{w}"
            if col not in window_df.columns:
                continue
            w_val = weights.get(w, 1.0)
            if weighted_sum is None:
                weighted_sum = window_df[col] * w_val
            else:
                weighted_sum = weighted_sum + window_df[col] * w_val
            active_weight += w_val

        if weighted_sum is not None and active_weight > 0:
            out[f"prob_{safe}"] = weighted_sum / active_weight
        else:
            out[f"prob_{safe}"] = np.nan

    # Normalise row-wise
    prob_cols = [f"prob_{r.replace(' ', '_')}" for r in REGIME_NAMES]
    prob_df   = out[prob_cols]
    row_sums  = prob_df.sum(axis=1).replace(0, np.nan)
    for col in prob_cols:
        out[col] = out[col] / row_sums

    # Primary = argmax across regime probability columns
    # Use numpy to handle all-NaN rows gracefully
    prob_arr = prob_df.values.astype(float)
    regime_arr = np.array(REGIME_NAMES)
    primary_vals = []
    for row in prob_arr:
        if np.all(np.isnan(row)):
            primary_vals.append(None)
        else:
            primary_vals.append(regime_arr[np.nanargmax(row)])
    out["regime_primary"] = primary_vals

    # --- Confidence layer 1: magnitude (mean across windows) -----------------
    mag_cols = [f"confidence_magnitude_{w}" for w in windows
                if f"confidence_magnitude_{w}" in window_df.columns]
    out["confidence_magnitude"] = window_df[mag_cols].mean(axis=1) if mag_cols else 0.0

    # --- Confidence layer 2: secondary (mean across windows) -----------------
    sec_cols = [f"confidence_secondary_{w}" for w in windows
                if f"confidence_secondary_{w}" in window_df.columns]
    out["confidence_secondary"] = window_df[sec_cols].mean(axis=1) if sec_cols else 0.5

    # --- Confidence layer 3: window consensus --------------------------------
    regime_cols = [f"regime_{w}" for w in windows if f"regime_{w}" in window_df.columns]
    if regime_cols:
        reg_df = window_df[regime_cols].copy()
        # For each row, count the most common non-null value
        def _row_consensus(row):
            valid = [v for v in row if v not in (None, "nan", "None", float("nan"))
                     and str(v) not in ("nan", "None")]
            if not valid:
                return 0.0
            most_common = max(set(valid), key=valid.count)
            return sum(1 for v in valid if v == most_common) / len(valid)
        out["confidence_consensus"] = reg_df.apply(_row_consensus, axis=1)
    else:
        out["confidence_consensus"] = 0.0

    # --- Combined confidence -------------------------------------------------
    out["regime_confidence"] = (
        out["confidence_magnitude"] +
        out["confidence_secondary"] +
        out["confidence_consensus"]
    ) / 3

    # Round confidence columns
    for col in ["confidence_magnitude", "confidence_secondary",
                "confidence_consensus", "regime_confidence"]:
        out[col] = out[col].round(4)

    # --- Transition flag -----------------------------------------------------
    low_consensus = out["confidence_consensus"] < config.TRANSITION_CONSENSUS_THRESHOLD
    label_changed = out["regime_primary"] != out["regime_primary"].shift(2)
    out["regime_transition"] = low_consensus | label_changed

    # --- Per-window smoothed labels (pass through for display) ---------------
    for w in windows:
        col = f"regime_{w}"
        if col in window_df.columns:
            out[col] = window_df[col]

    return out


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def classify_regimes(
    factors_df: pd.DataFrame,
    config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full classification pipeline.

    Parameters
    ----------
    factors_df : output of factor engine (compute_factors)
    config     : config module

    Returns
    -------
    ensemble_df  : pd.DataFrame — primary output per date
                   regime_primary, regime_confidence, prob_*, transition flag,
                   per-window smoothed labels
    window_df    : pd.DataFrame — full per-window detail (for debugging/display)
    """
    logger.info("Running regime classifier...")

    window_df   = classify_all_windows(factors_df, config)
    ensemble_df = compute_ensemble(window_df, factors_df, config)

    # Summary log
    if not ensemble_df.empty and "regime_primary" in ensemble_df.columns:
        valid = ensemble_df.dropna(subset=["regime_primary"])
        if not valid.empty:
            latest = valid.iloc[-1]
            logger.info(
                f"Classifier complete. Latest: {latest['regime_primary']} "
                f"(confidence={latest['regime_confidence']:.2f}, "
                f"transition={latest['regime_transition']})"
            )
        else:
            logger.warning("Classifier: ensemble produced no valid regime calls — "
                           "check factor engine output for NaNs")

    return ensemble_df, window_df


# =============================================================================
# CONVENIENCE ACCESSORS
# =============================================================================

def get_current_regime(ensemble_df: pd.DataFrame) -> dict:
    """Return the most recent regime call as a plain dict."""
    if ensemble_df.empty:
        return {}
    valid = ensemble_df.dropna(subset=["regime_primary"])
    if valid.empty:
        return {}
    return valid.iloc[-1].to_dict()


def get_regime_history(
    ensemble_df: pd.DataFrame,
    config,
    smoothed: bool = True,
) -> pd.Series:
    """
    Return the primary regime label series over time.
    smoothed=True uses the majority-vote smoothed label.
    """
    col = "regime_primary"
    if col not in ensemble_df.columns:
        return pd.Series(dtype=str)
    return ensemble_df[col].dropna()


def get_probability_history(ensemble_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the full probability matrix over time.
    Columns: prob_{regime} for each of five regimes.
    """
    prob_cols = [c for c in ensemble_df.columns if c.startswith("prob_")
                 and not any(w in c for w in ["36m", "60m", "120m", "240m", "expanding"])]
    return ensemble_df[prob_cols].dropna(how="all")
