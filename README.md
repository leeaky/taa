# TAA Regime Model

A systematic global macro regime model for Tactical Asset Allocation, built on Fullerton Fund Management's four-factor Investment Environment framework. Classifies the global economy into five regimes using z-score deviations from trend across growth, inflation, liquidity, and risk appetite — and maps historical asset class performance to each regime.

Built in Python. Runs daily. Designed for discretionary portfolio managers who want a systematic signal to inform, not replace, human judgment.

---

## What this is

Most TAA frameworks pick a window, pick a regime label, and present a single number as fact. This model doesn't do that.

Instead it:
- Computes every factor score across five lookback windows simultaneously (3yr, 5yr, 10yr, 20yr, full history)
- Outputs a **probability vector** across five regimes rather than a hard label
- Flags when windows disagree — because disagreement is itself a signal
- Tracks what it said and when via an append-only state log, so the live history is epistemically honest
- Treats data revisions as new information arriving today, not corrections to the past

The discretionary overlay does real work here. The model tells you where you are and how confident it is. The portfolio manager decides what to do about it.

---

## Framework

Inspired by and modelled on Fullerton Fund Management's Investment Environment Model, as disclosed across their quarterly Investment Views (FIV) reports Q1 2023–Q2 2026.

**Four factors, each expressed as z-score deviations from trend:**

| Factor | What it measures |
|--------|-----------------|
| Growth | Leading (yield curve, OECD CLI, equity momentum) + Coincident (GDP trend deviation) |
| Inflation | CPI, PCE, 5yr5yr forward inflation expectations, TIPS real yield, commodity momentum |
| Liquidity | M2 growth, Fed balance sheet, real rate level, USD direction |
| Risk Appetite | VIX, put/call ratio, HY credit spreads, EM/DM relative performance, equity trend |

**Five regimes, output as probability distribution:**

| Regime | Growth | Inflation | Character |
|--------|--------|-----------|-----------|
| Recovery / Early Cycle | ↑ | ↓ | Growth recovering, inflation contained |
| Goldilocks | + | − | Above-trend growth, below-trend inflation |
| Late Cycle | ↓ | ↑ | Decelerating growth, rising inflation |
| Danger Zone | − | + | Falling growth, elevated stress |
| Sentiment Driven | ~ | ~ | Macro ambiguous — risk appetite dominant |

Sentiment Driven is treated as a distinct positive regime (consistent with Fullerton's published framework), not merely an uncertainty flag.

**Confidence scoring — three layers:**
1. Factor magnitude — how far are Growth and Inflation from zero
2. Secondary confirmation — do Liquidity and Risk Appetite support the regime call
3. Window consensus — what fraction of the five window variants agree

---

## Architecture

```
src/
├── config.py               # All parameters — tickers, windows, thresholds, lags, regime ideals
├── data/
│   ├── fetcher.py          # Pull from FRED + Yahoo Finance, cache locally, incremental refresh
│   ├── vintage_manager.py  # Lag offsets, staleness flagging, revision signals
│   ├── state_log.py        # Append-only record of every model run
│   └── synthetic.py        # Regime-aware synthetic data for testing/showcase
├── factors/
│   └── engine.py           # Four-factor computation across all z-score windows + consensus
├── regimes/
│   └── classifier.py       # Five-regime probability vector, confidence scoring, smoothing
├── analysis/
│   └── regime_map.py       # Historical asset class performance by regime
├── dashboard/
│   └── app.py              # Streamlit dashboard — five-page live presentation layer
└── tests/
    ├── test_module1.py     # Data fetcher + vintage manager
    ├── test_module2.py     # Factor engine
    ├── test_module3.py     # Regime classifier
    └── test_module4.py     # Historical regime map
```

---

## Key design decisions

**Config-driven.** Every meaningful number lives in `config.py` — z-score windows, publication lag offsets, regime ideal points, softmax temperature, momentum windows, staleness grace periods. Nothing is hardcoded in logic modules. Calibrating the model on live data means touching only `config.py`.

**Data vintaging.** The model only uses data that was publicly available at the time of each run. Publication lags are encoded per series in config. Staleness is flagged visibly, never silently filled. Revisions are treated as new information arriving today, not corrections to the past.

**Window parameterisation.** Every factor is computed across five lookback windows simultaneously. A 3-year window asks "how does today compare to recent history." A 20-year window asks "how does today compare to a full cycle." These are different questions and will sometimes give different answers — that disagreement is surfaced, not hidden. Where windows diverge, confidence scores fall and transition flags fire.

**Probabilistic output.** Rather than a single hard regime label, the classifier outputs a probability vector across all five regimes at every date — consistent with Fullerton's disclosed Investment Environment Indicator. The primary label is the highest-probability regime, but the full distribution is always visible.

**State log.** Every daily run appends one row recording what the model said and what data it used. This is the live epistemic record. The historical bootstrap (before day one of live running) uses revised public data and is clearly labelled as illustrative. BOOTSTRAP and LIVE periods are tagged separately throughout.

**Daily cadence with incremental refresh.** The fetcher checks what's changed since the last run and pulls only updated series. Most days nothing changes. On data release days, the model updates immediately and logs what changed and why.

**Discretionary overlay.** The model produces a signal. A human decides what to do with it. This is by design — the systematic layer compensates for cognitive bias and data overload; the discretionary layer compensates for model blind spots and regime transitions.

---

## Data sources

| Source | Series | Tier |
|--------|--------|------|
| FRED | GDP, CPI, PCE, M2, Fed balance sheet, yield curve, TIPS, forward inflation, OECD CLI | REVISED* |
| Yahoo Finance | VIX, put/call ratio, S&P 500, global equity ETFs, credit ETFs, commodities, gold | REALTIME |

`*REVISED` = standard FRED data — retrospectively clean but not what existed in real time. Flagged in dashboard. ALFRED vintage data or Bloomberg DDIS would close this gap.

**Bloomberg gap:** The liquidity factor is the weakest on public data. ECB, PBoC, and BOJ balance sheet data, along with global M2 composites, require Bloomberg or CEIC. The current proxy (Fed-centric with USD as a global tightening proxy) is flagged in the dashboard and will be replaced when Bloomberg access is available. Only the fetcher and config need to change — nothing else.

---

## Build status

| Module | Status | Notes |
|--------|--------|-------|
| `config.py` | ✅ Complete | Five-regime taxonomy, all tickers, lags, regime ideals, calibration parameters |
| `data/fetcher.py` | ✅ Complete | FRED + Yahoo, incremental refresh, local cache |
| `data/vintage_manager.py` | ✅ Complete | Lag offsets, staleness flagging, revision + surprise signals |
| `data/state_log.py` | ✅ Complete | Append-only, records every daily run |
| `data/synthetic.py` | ✅ Complete | Regime-aware synthetic data, exact schema match to live fetcher |
| `factors/engine.py` | ✅ Complete | All four factors, five windows, consensus scoring |
| `regimes/classifier.py` | ✅ Complete | Five-regime probability vector, three-layer confidence, smoothing |
| `analysis/regime_map.py` | ✅ Complete | Regime periods, asset return stats, three summary matrices |
| `dashboard/app.py` | ✅ Complete | Five-page Streamlit dashboard, dark terminal aesthetic |
| `tests/test_module*.py` | ✅ Complete | Independent test scripts for each module |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run all module tests (synthetic data, no internet required)
cd C:\dev\taa
python src/tests/test_module1.py
python src/tests/test_module2.py
python src/tests/test_module3.py
python src/tests/test_module4.py

# Launch dashboard — synthetic data
streamlit run src/dashboard/app.py

# Launch dashboard — live data (toggle in sidebar, requires internet)
streamlit run src/dashboard/app.py
```

To switch from synthetic to live data: toggle "Live data" in the dashboard sidebar, or set `USE_LIVE_DATA = True` in any test script.

---

## Calibration notes

The model ships with sensible defaults but several parameters will benefit from calibration once live data accumulates:

**`SOFTMAX_TEMPERATURE`** (default 1.5) — controls how sharply the probability distribution concentrates around the nearest regime. Lower = more decisive calls. If Sentiment Driven dominates, reduce this toward 1.0.

**`REGIME_IDEALS`** — the (growth, inflation) ideal point for each regime in z-score space. These encode where in factor space each regime sits. Review against actual factor readings once 6–12 months of live data exists.

**`REGIME_SECONDARY_IDEALS`** — expected (liquidity, risk appetite) signatures per regime. Used for confidence layer 2. Same calibration timing as above.

**`SENTIMENT_RA_THRESHOLD` / `SENTIMENT_BOOST_WEIGHT`** — govern how strongly elevated risk appetite boosts the Sentiment Driven probability when macro factors are ambiguous.

All in `config.py`. No code changes required to calibrate.

---

## Roadmap

- [x] Module 1: Data ingestion — FRED + Yahoo, vintaging, state log
- [x] Module 2: Factor engine — four factors, five windows, consensus
- [x] Module 3: Regime classifier — five-regime probability vector
- [x] Module 4: Historical regime map — asset class performance by regime
- [x] Module 5: Streamlit dashboard — live presentation layer
- [ ] Bloomberg integration — replace thin liquidity proxies with full global CB coverage
- [ ] ALFRED vintage data — close the real-time data gap for bootstrap history
- [ ] Regional expansion — US, Europe, Asia roll-up to global composite
- [ ] Augmented intelligence layer — alternative data, NLP sentiment signals
- [ ] Portfolio construction layer — regime-conditional position sizing and risk budgeting

---

## Reference

Fullerton Fund Management, Fullerton Investment Views (FIV), Q1 2023–Q2 2026.
All factor construction and regime taxonomy is inferred from public disclosures.
Fullerton's internal weighting methodology and aggregation rules are proprietary and not replicated here.

---

*This model is a systematic decision-support tool, not investment advice. Past regime-conditional returns are descriptive, not predictive.*
