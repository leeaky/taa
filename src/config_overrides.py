# =============================================================================
# CONFIG OVERRIDE LOADER
# Loads config.py as the base, then applies any overrides from
# config_overrides.json if it exists. The JSON file is written by the
# dashboard config editor and never committed to git.
#
# Usage: import config_overrides as config
# (drop-in replacement for: import config)
#
# Workflow:
#   1. Tweak parameters in the dashboard Config Editor page
#   2. Model reruns with overrides applied
#   3. When satisfied, use "Promote to defaults" to update config.py
#   4. Delete config_overrides.json to reset to defaults
# =============================================================================

import json
import copy
from pathlib import Path

# --- Load base config --------------------------------------------------------
import config as _base

# Expose everything from base config by default
import sys
_this = sys.modules[__name__]
for _k in dir(_base):
    if not _k.startswith("__"):
        setattr(_this, _k, getattr(_base, _k))

# --- Override file path ------------------------------------------------------
OVERRIDES_PATH = Path(_base.BASE_DIR) / "config_overrides.json"

# --- Editable parameter definitions ------------------------------------------
# These are the parameters exposed in the config editor.
# Format: { key: { "type": ..., "label": ..., "group": ..., "help": ... } }
EDITABLE_PARAMS = {
    # Window weights
    "ZSCORE_WINDOW_WEIGHTS": {
        "type": "window_weights",
        "group": "Window Weights",
        "label": "Window Weights for Ensemble",
        "help": "Relative weight of each z-score window in the ensemble probability average. "
                "Downweight 240m and expanding to reduce long-run anchor bias "
                "and Sentiment Driven dominance.",
    },
    # Classifier
    "SOFTMAX_TEMPERATURE": {
        "type": "float", "min": 0.5, "max": 3.0, "step": 0.1,
        "group": "Classifier",
        "label": "Softmax Temperature",
        "help": "Controls sharpness of regime probability distribution. "
                "Lower = more decisive calls. Higher = softer, more spread. "
                "Reduce toward 1.0 if Sentiment Driven dominates.",
    },
    "SMOOTHING_WINDOW": {
        "type": "int", "min": 1, "max": 6,
        "group": "Classifier",
        "label": "Smoothing Window (months)",
        "help": "Rolling mode window to suppress whipsawing between regimes.",
    },
    "TRANSITION_CONSENSUS_THRESHOLD": {
        "type": "float", "min": 0.4, "max": 0.9, "step": 0.05,
        "group": "Classifier",
        "label": "Transition Consensus Threshold",
        "help": "Below this fraction of windows agreeing, a transition flag fires.",
    },
    # Sentiment Driven
    "SENTIMENT_RA_THRESHOLD": {
        "type": "float", "min": 0.0, "max": 1.5, "step": 0.05,
        "group": "Sentiment Driven Regime",
        "label": "Risk Appetite Threshold",
        "help": "Risk appetite must exceed this to trigger Sentiment Driven boost.",
    },
    "SENTIMENT_BOOST_WEIGHT": {
        "type": "float", "min": 0.0, "max": 1.5, "step": 0.05,
        "group": "Sentiment Driven Regime",
        "label": "Sentiment Boost Weight",
        "help": "Multiplier on the Sentiment Driven boost signal.",
    },
    # Growth factor
    "GROWTH_LEAD_WEIGHT": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
        "group": "Growth Factor",
        "label": "Leading Indicator Weight",
        "help": "Weight on leading sub-components (yield curve, CLI, equity momentum). "
                "Coincident weight = 1 - this value.",
    },
    # Windows
    "MOMENTUM_WINDOW_SHORT": {
        "type": "int", "min": 1, "max": 6,
        "group": "Calculation Windows",
        "label": "Momentum Window (months)",
        "help": "Lookback for commodity, USD, EM/DM momentum.",
    },
    "EQUITY_MA_WINDOW": {
        "type": "int", "min": 3, "max": 24,
        "group": "Calculation Windows",
        "label": "Equity MA Window (months)",
        "help": "Moving average window for equity trend signal.",
    },
    "YOY_WINDOW": {
        "type": "int", "min": 6, "max": 24,
        "group": "Calculation Windows",
        "label": "Year-on-Year Window (months)",
        "help": "Window for YoY calculations (CPI, PCE, M2, etc).",
    },
    # Regime ideals — Growth axis
    "REGIME_IDEALS": {
        "type": "regime_ideals",
        "group": "Regime Anchor Points",
        "label": "Regime Ideal Points (Growth, Inflation)",
        "help": "Where each regime sits in (growth z-score, inflation z-score) space. "
                "Most important parameters to calibrate on live data.",
    },
    "REGIME_SECONDARY_IDEALS": {
        "type": "regime_secondary",
        "group": "Regime Anchor Points",
        "label": "Secondary Ideals (Liquidity, Risk Appetite)",
        "help": "Expected (liquidity, risk appetite) signature per regime. "
                "Used for confidence layer 2.",
    },
}


# --- Load and apply overrides ------------------------------------------------

def load_overrides() -> dict:
    """Load overrides from JSON file. Returns empty dict if file doesn't exist."""
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        with open(OVERRIDES_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not load config_overrides.json: {e}")
        return {}


def save_overrides(overrides: dict):
    """Save overrides dict to JSON file."""
    with open(OVERRIDES_PATH, "w") as f:
        json.dump(overrides, f, indent=2)


def delete_overrides():
    """Delete the overrides file — resets to defaults."""
    if OVERRIDES_PATH.exists():
        OVERRIDES_PATH.unlink()


def get_current_value(key: str, overrides: dict):
    """Get the current effective value for a key (override if present, else base)."""
    if key in overrides:
        return overrides[key]
    return getattr(_base, key, None)


def get_diff(overrides: dict) -> dict:
    """Return dict of params that differ from base config defaults."""
    diff = {}
    for key, val in overrides.items():
        base_val = getattr(_base, key, None)
        if val != base_val:
            diff[key] = {"override": val, "default": base_val}
    return diff


def apply_overrides_to_module(overrides: dict):
    """Apply override values to this module's namespace."""
    for key, val in overrides.items():
        if hasattr(_base, key):
            setattr(_this, key, val)


def promote_to_defaults(overrides: dict) -> str:
    """
    Write overridden values back into config.py as new defaults.
    Returns a summary of what was changed.

    This modifies config.py in place. The override file is NOT deleted —
    call delete_overrides() separately after reviewing.
    """
    config_path = Path(_base.BASE_DIR) / "config.py"
    with open(config_path, "r") as f:
        src = f.read()

    changes = []
    for key, new_val in overrides.items():
        old_val = getattr(_base, key, None)
        if new_val == old_val:
            continue

        # Only handle simple scalar types (not dicts — those need manual review)
        if isinstance(new_val, (int, float, bool)):
            # Find and replace the assignment line
            import re
            pattern = rf"^({re.escape(key)}\s*=\s*)(.+?)(\s*#.*)?"
            old_line = None
            new_line = None
            for line in src.splitlines():
                m = re.match(pattern, line.strip())
                if m and line.strip().startswith(key):
                    old_line = line
                    comment = m.group(3) or ""
                    new_line = line.replace(
                        m.group(2).strip(),
                        f"{new_val:.4f}" if isinstance(new_val, float) else str(new_val)
                    )
                    break

            if old_line and new_line and old_line != new_line:
                src = src.replace(old_line, new_line, 1)
                changes.append(f"  {key}: {old_val} → {new_val}")

        elif isinstance(new_val, dict):
            changes.append(f"  {key}: dict — update config.py manually")

    with open(config_path, "w") as f:
        f.write(src)

    return "\n".join(changes) if changes else "No scalar changes to write."


# Apply overrides on import
# NOTE: We do NOT apply overrides at module level here.
# Instead, the dashboard reads fresh overrides on each Streamlit rerun
# via get_effective_config(), avoiding Streamlit's module cache problem.

def get_effective_config() -> dict:
    """
    Return a dict of all config values with overrides applied.
    Always reads from disk — never relies on cached module state.
    Call this instead of reading config attributes directly when you
    need override-aware values at runtime.
    """
    overrides = load_overrides()
    result = {}
    for key in dir(_base):
        if not key.startswith("__"):
            result[key] = overrides.get(key, getattr(_base, key))
    return result
