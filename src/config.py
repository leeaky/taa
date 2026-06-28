# =============================================================================
# TAA MODEL — CONFIGURATION
# All tickers, windows, thresholds, and lags live here.
# Nothing is hardcoded in logic modules.
# =============================================================================

from pathlib import Path

# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "cache"
LOG_DIR  = BASE_DIR / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

STATE_LOG_PATH    = LOG_DIR / "state_log.csv"
DATA_QUALITY_LOG  = LOG_DIR / "data_quality.log"
LAST_RUN_PATH     = LOG_DIR / "last_run.json"

# --- Date Range --------------------------------------------------------------
HISTORY_START = "1990-01-01"   # how far back to pull raw data

# --- Z-Score Windows (months) ------------------------------------------------
ZSCORE_WINDOWS   = [36, 60, 120, 240]   # 3yr, 5yr, 10yr, 20yr
ZSCORE_EXPANDING = True                  # full history as fifth variant

# --- Factor Engine -----------------------------------------------------------
GROWTH_LEAD_WEIGHT    = 0.60    # leading vs coincident split
GROWTH_COIN_WEIGHT    = 0.40
FACTOR_EQUAL_WEIGHT   = True    # sub-components equal weighted within each factor

# Revision momentum weight within factor score (small — directional signal only)
REVISION_WEIGHT       = 0.10
SURPRISE_WEIGHT       = 0.05

# Sub-component calculation windows (months)
MOMENTUM_WINDOW_SHORT = 3       # commodity, USD, EM/DM momentum lookback
EQUITY_MA_WINDOW      = 12      # equity trend moving average
YOY_WINDOW            = 12      # year-on-year calculation window
GDP_TREND_WINDOW      = 120     # rolling window for GDP trend (10yr)
GDP_TREND_MIN_PERIODS = 24      # minimum months before GDP trend computed
ZSCORE_MIN_PERIODS    = 36      # minimum months before expanding z-score computed

# --- Regime Classifier -------------------------------------------------------
SMOOTHING_WINDOW               = 3     # months, rolling mode
TRANSITION_CONSENSUS_THRESHOLD = 0.60  # below this → transition flagged
CONFIDENCE_LAYERS_EQUAL_WEIGHT = True
STALENESS_GRACE_DAYS           = 7     # days overdue before series flagged stale

# Softmax temperature — controls sharpness of probability distribution.
# Lower = more decisive (spikier) calls. Higher = softer, more spread.
# 1.0 = sharp. 2.0 = soft. Calibrate on live data if Sentiment Driven dominates.
SOFTMAX_TEMPERATURE   = 1.5

# Sentiment Driven boost parameters
# Applied when macro is ambiguous (growth and inflation near zero)
# and risk appetite is elevated — consistent with Fullerton's treatment
# of Sentiment Driven as a distinct positive regime.
SENTIMENT_RA_THRESHOLD   = 0.30   # risk appetite must exceed this to trigger boost
SENTIMENT_BOOST_WEIGHT   = 0.50   # multiplier on the boost signal

# Regime ideal points — (growth_zscore, inflation_zscore) for each regime.
# These define where in factor space each regime sits.
# MOST IMPORTANT numbers to calibrate on live data.
# Based on Fullerton's disclosed regime characteristics.
REGIME_IDEALS = {
    "Recovery":         ( 0.5, -1.0),
    "Goldilocks":       ( 1.2, -0.5),
    "Late Cycle":       ( 0.3,  1.0),
    "Danger Zone":      (-1.2,  0.5),
    "Sentiment Driven": ( 0.0,  0.0),
}

# Secondary factor signatures — expected (liquidity, risk_appetite) per regime.
# Used for confidence layer 2.
REGIME_SECONDARY_IDEALS = {
    "Recovery":         ( 0.5,  0.0),
    "Goldilocks":       ( 0.3,  0.8),
    "Late Cycle":       (-0.3,  0.2),
    "Danger Zone":      (-0.8, -1.0),
    "Sentiment Driven": ( 0.0,  0.5),
}

# --- Release Calendar --------------------------------------------------------
# Typical business days after reference period end before data is published.
# Conservative (late) estimates — we'd rather wait than use stale data.
SERIES_LAGS_DAYS = {
    "CPIAUCSL":    45,   # CPI — mid month following reference month
    "PCEPI":       35,   # PCE
    "GDP":         30,   # GDP first release (quarterly)
    "OECD_CLI":    45,   # OECD Composite Leading Indicator
    "M2SL":         7,   # M2 money supply — weekly, ~1 week lag
    "WALCL":        7,   # Fed balance sheet — weekly H.4.1
    "T10Y2Y":       0,   # 10y-2y yield spread — daily market
    "DFII5":        0,   # 5yr TIPS real yield — daily
    "T5YIFR":       0,   # 5yr5yr forward inflation — daily
    "DTWEXBGS":     1,   # USD broad index — 1 day lag
    "BAMLH0A0HYM2": 1,   # HY credit spread — 1 day lag
    "^VIX":         0,   # VIX — real time
    "^PCALL":       0,   # CBOE total put/call ratio — real time (Fullerton confirmed)
    "ISM_PMI":      3,   # ISM PMI proxy — first business day
}

# Data tier labels (for dashboard transparency)
# REALTIME           = market prices, no revision
# VINTAGE_APPROXIMATE = ALFRED where available
# REVISED            = standard FRED, retrospectively clean
DATA_TIERS = {
    "CPIAUCSL":    "REVISED",
    "PCEPI":       "REVISED",
    "GDP":         "REVISED",
    "OECD_CLI":    "REVISED",
    "M2SL":        "REVISED",
    "WALCL":       "REALTIME",
    "T10Y2Y":      "REALTIME",
    "DFII5":       "REALTIME",
    "T5YIFR":      "REALTIME",
    "DTWEXBGS":    "REALTIME",
    "BAMLH0A0HYM2":"REALTIME",
    "^VIX":        "REALTIME",
    "^PCALL":      "REALTIME",
    "ISM_PMI":     "REVISED",
}

# --- FRED Series -------------------------------------------------------------
FRED_SERIES = {
    # Growth — coincident
    "GDP":         "US GDP (quarterly, billions USD)",
    # Growth — leading
    "T10Y2Y":      "10y-2y yield spread (leading indicator)",
    "OECD_CLI":    "OECD CLI — proxied via FRED USALOLITONOSTSAM",
    # Inflation
    "CPIAUCSL":    "CPI All Urban Consumers YoY",
    "PCEPI":       "PCE Price Index",
    "T5YIFR":      "5yr5yr forward inflation expectation",
    "DFII5":       "5yr TIPS real yield",
    # Liquidity
    "M2SL":        "M2 Money Supply",
    "WALCL":       "Fed Balance Sheet (total assets)",
    "DTWEXBGS":    "USD Broad Index",
    # Credit / Risk Appetite
    "BAMLH0A0HYM2":"ICE BofA HY OAS spread",
}

# OECD CLI actual FRED series ID
FRED_OECD_CLI_ID = "USALOLITONOSTSAM"

# --- Yahoo Finance Tickers ---------------------------------------------------
YAHOO_TICKERS = {
    # Risk appetite / factor inputs
    # Fullerton confirmed: VIX + put/call ratio are explicit components
    "^VIX":   "CBOE VIX (fear index — inverted for risk appetite)",
    "^PCALL": "CBOE Total Put/Call Ratio (inverted for risk appetite — Fullerton confirmed)",
    "^GSPC":  "S&P 500 (equity momentum / growth coincident)",
    # Asset class proxies for regime map
    # Aligned to Fullerton's disclosed return distribution assets:
    # US equities (^GSPC above), DM equities (VT), EM equities (EEM),
    # US 10y bonds (TLT), USD (DTWEXBGS in FRED), Commodities (DJP), Gold (GLD)
    # Plus credit proxies (HYG, LQD) for factor inputs
    "VT":     "Global Equities / DM proxy (Vanguard Total World)",
    "AGG":    "US Aggregate Bonds",
    "EEM":    "EM Equities",
    "HYG":    "High Yield Credit (risk appetite input + regime map)",
    "LQD":    "IG Credit",
    "DJP":    "Commodities (Bloomberg Commodity proxy)",
    "GLD":    "Gold",
    "TLT":    "Long Duration US Treasuries",
}

# Asset classes used in regime map
# Aligned to Fullerton's disclosed regime return distributions:
# US Eq (^GSPC), DM Eq (VT), EM Eq (EEM), US 10y bonds (TLT),
# Commodities (DJP), Gold (GLD) — USD covered via DTWEXBGS in FRED
# Credit proxies (HYG, AGG) added for fuller picture
REGIME_MAP_ASSETS = ["^GSPC", "VT", "EEM", "TLT", "AGG", "HYG", "DJP", "GLD"]

# --- Regime Labels -----------------------------------------------------------
# Five-regime taxonomy confirmed across all Fullerton FIV reports (Q1 2023–Q2 2026).
# Labels vary slightly ("Recovery" vs "Early Cycle") — both included as aliases.
# Primary classification axes: Growth (primary) + Inflation (primary)
# Secondary modifiers:         Liquidity + Risk Appetite → confidence + regime tilt
#
# Regime characteristics (for classifier logic):
#   Recovery/Early Cycle : growth recovering from below trend, inflation low/falling
#   Goldilocks           : growth above trend, inflation below trend — best for risk
#   Late Cycle           : growth still positive but decelerating, inflation rising,
#                          liquidity tightening, risk appetite fading
#   Danger Zone          : growth falling sharply, stress in financial conditions
#   Sentiment Driven     : macro factors ambiguous/conflicting, market driven by
#                          positioning and mood — risk appetite is the dominant signal
#                          (Fullerton treats this as a distinct positive regime, not
#                          merely an uncertainty flag)
REGIME_LABELS = {
    "Recovery":        {"growth": "↑", "inflation": "↓", "color": "#27ae60",
                        "alias": "Early Cycle",
                        "description": "Growth recovering, inflation contained"},
    "Goldilocks":      {"growth": "+", "inflation": "-", "color": "#2ecc71",
                        "alias": None,
                        "description": "Above-trend growth, below-trend inflation"},
    "Late Cycle":      {"growth": "↓", "inflation": "↑", "color": "#f39c12",
                        "alias": None,
                        "description": "Decelerating growth, rising inflation"},
    "Danger Zone":     {"growth": "-", "inflation": "+", "color": "#e74c3c",
                        "alias": None,
                        "description": "Falling growth, elevated stress"},
    "Sentiment Driven":{"growth": "~", "inflation": "~", "color": "#9b59b6",
                        "alias": None,
                        "description": "Macro ambiguous — risk appetite dominant signal"},
}

# --- Dashboard ---------------------------------------------------------------
DASHBOARD_TITLE      = "TAA Regime Model"
REFRESH_CACHE_HOURS  = 24     # how old cache can be before forced refresh
