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
| Risk Appetite | VIX, HY credit spreads, EM/DM relative performance, equity trend |

**Five regimes, output as probability distribution:**

| Regime | Growth | Inflation | Character |
|--------|--------|-----------|-----------|
| Recovery / Early Cycle | ↑ | ↓ | Growth recovering, inflation contained |
| Goldilocks | + | − | Above-trend growth, below-trend inflation |
| Late Cycle | ↓ | ↑ | Decelerating growth, rising inflation |
| Danger Zone | − | + | Falling growth, elevated stress |
| Sentiment Driven | ~ | ~ | Macro ambiguous — risk appetite dominant |

Sentiment Driven is treated as a distinct positive regime (consistent with Fullerton's published framework), not merely an uncertainty flag. When macro factors are near trend and risk appetite is elevated, positioning and mood drive returns more than fundamentals.

**Confidence scoring — three layers:**
1. Factor magnitude — how far are Growth and Inflation from zero
2. Secondary confirmation — do Liquidity and Risk Appetite support the primary call
3. Window consensus — what fraction of the five window variants agree

**Window weighting.** Each window gets a configurable weight in the ensemble probability average. Shorter windows (3yr, 5yr) reflect the current cycle; longer windows (20yr, full history) provide structural context. Default: shorter windows weighted equally at 1.0, longer windows downweighted to 0.5. Adjustable in the Config Editor without touching code.

---

## Architecture

```
src/
├── config.py                # All parameters — tickers, windows, weights, thresholds, regime ideals
├── config_overrides.py      # Override layer — reads config_overrides.json, applied at runtime
├── runner.py                # Daily runner — fetches data, runs pipeline, writes state log
├── data/
│   ├── fetcher.py           # Pull from FRED + Yahoo Finance, cache locally, incremental refresh
│   ├── vintage_manager.py   # Lag offsets, staleness flagging, revision signals
│   ├── state_log.py         # Append-only record of every model run
│   └── synthetic.py         # Regime-aware synthetic data for testing/showcase
├── factors/
│   └── engine.py            # Four-factor computation across all z-score windows + consensus
├── regimes/
│   └── classifier.py        # Five-regime probability vector, weighted ensemble, confidence scoring
├── analysis/
│   └── regime_map.py        # Historical asset class performance by regime
├── dashboard/
│   └── app.py               # Streamlit dashboard — six-page live presentation layer
└── tests/
    ├── test_module1.py      # Data fetcher + vintage manager
    ├── test_module2.py      # Factor engine
    ├── test_module3.py      # Regime classifier
    ├── test_module4.py      # Historical regime map
    └── diagnose_live.py     # Live data diagnostic — run when data issues arise
```

---

## Key design decisions

**Config-driven.** Every meaningful number lives in `config.py` — z-score windows, window weights, publication lag offsets, regime ideal points, softmax temperature, momentum windows, staleness grace periods. Nothing is hardcoded in logic modules. Calibrating the model means touching only `config.py` or using the dashboard Config Editor.

**Override system.** `config_overrides.json` sits alongside `config.py` and layers parameter overrides on top without modifying the base config. Saved from the dashboard Config Editor. Delete the file to reset to defaults. Use "Promote to config.py" when a calibrated set is ready to become the new default.

**Data vintaging.** The model only uses data that was publicly available at the time of each run. Publication lags are encoded per series in config. Staleness is flagged visibly, never silently filled. Revisions are treated as new information arriving today, not corrections to the past.

**Window parameterisation.** Every factor is computed across five lookback windows simultaneously. A 3-year window asks "how does today compare to recent history." A 20-year window asks "how does today compare to a full cycle." These are different questions and will sometimes give different answers — that disagreement is surfaced, not hidden. Window weights control how much each lookback influences the primary regime call.

**Probabilistic output.** Rather than a single hard regime label, the classifier outputs a probability vector across all five regimes at every date. The primary label is the highest-probability regime, but the full distribution is always visible — consistent with Fullerton's disclosed Investment Environment Indicator.

**State log.** Every daily run appends one row recording what the model said and what data it used. This is the live epistemic record. The historical bootstrap (before day one of live running) uses revised public data and is clearly labelled as illustrative. BOOTSTRAP and LIVE periods are tagged separately throughout. The state log is the difference between a track record and a backtest.

**Daily cadence with incremental refresh.** The runner checks what's changed since the last run and pulls only updated series. Most days nothing changes. On data release days, the model updates and logs what changed and why.

**Discretionary overlay.** The model produces a signal. A human decides what to do with it. The systematic layer compensates for cognitive bias and data overload; the discretionary layer compensates for model blind spots and regime transitions.

---

## Data sources

| Source | Series | Tier |
|--------|--------|------|
| FRED | GDP, CPI, PCE, M2, Fed balance sheet, yield curve, TIPS, forward inflation, OECD CLI | REVISED* |
| Yahoo Finance | VIX, S&P 500, global equity ETFs, credit ETFs, commodities, gold | REALTIME |

`*REVISED` = standard FRED data — retrospectively clean but not what existed in real time. Flagged in dashboard. ALFRED vintage data or Bloomberg DDIS would close this gap.

**Known data gaps:**
- OECD CLI runs ~2.5 years behind on FRED — growth factor uses yield curve, equity momentum, and GDP as compensating sub-components
- Put/call ratio not available via Yahoo Finance — VIX carries this weight in the risk appetite factor
- Liquidity factor is Fed-centric — ECB, PBoC, BOJ balance sheet data requires Bloomberg or CEIC

---

## Build status

| File | Status | Notes |
|------|--------|-------|
| `config.py` | ✅ Live | Five-regime taxonomy, window weights, all calibration parameters |
| `config_overrides.py` | ✅ Live | Override layer with dashboard editor integration |
| `runner.py` | ✅ Live | Daily runner, state log writer, change detection |
| `data/fetcher.py` | ✅ Live | FRED + Yahoo, incremental refresh, local cache |
| `data/vintage_manager.py` | ✅ Live | Lag offsets, quarterly resampling, staleness flagging |
| `data/state_log.py` | ✅ Live | Append-only, auto-created on first run |
| `data/synthetic.py` | ✅ Live | Regime-aware synthetic data, exact schema match |
| `factors/engine.py` | ✅ Live | All four factors, five windows, consensus scoring |
| `regimes/classifier.py` | ✅ Live | Five-regime probability vector, weighted ensemble |
| `analysis/regime_map.py` | ✅ Live | Regime periods, asset return stats, BOOTSTRAP/LIVE tagging |
| `dashboard/app.py` | ✅ Live | Six-page Streamlit dashboard — Current State, Factor History, Regime History, Asset Performance, Data Quality, Config Editor |
| `tests/` | ✅ Live | Independent test + diagnostic scripts |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run all module tests (synthetic data, no internet required)
python src/tests/test_module1.py
python src/tests/test_module2.py
python src/tests/test_module3.py
python src/tests/test_module4.py

# Run daily model and write first state log entry (requires internet)
python src/runner.py

# Diagnose live data issues
python src/tests/diagnose_live.py

# Launch dashboard (keep terminal open while using)
streamlit run src/dashboard/app.py
```

To switch from synthetic to live data: toggle "Live data" in the dashboard sidebar.

---

## Daily operation

The runner is designed to be scheduled via Windows Task Scheduler:

```
Program:   python
Arguments: C:\dev\taa\src\runner.py
Start in:  C:\dev\taa
Trigger:   Daily at 08:00
```

Each run:
1. Fetches latest data from FRED and Yahoo Finance
2. Detects what changed since yesterday
3. Runs the full pipeline (vintage → factors → classifier)
4. Appends one row to `src/logs/state_log.csv`
5. Logs factor readings and regime call to `src/logs/runner.log`
6. Exits (~30 seconds total)

The dashboard reads from the cache and state log independently — it does not need to be running for the daily run to work.

**State log location:** `src/logs/state_log.csv` — created automatically on first run. Do not edit manually.

---

## Calibration

All calibration is done through the **Config Editor** page in the dashboard. Changes are saved to `config_overrides.json` and take effect immediately without restarting. Reset to defaults by clicking "Reset to defaults" or deleting the JSON file.

Key parameters to calibrate once 3–6 months of live data has accumulated:

**`SOFTMAX_TEMPERATURE`** — sharpness of regime probability distribution. Reduce toward 1.0 if Sentiment Driven dominates.

**`ZSCORE_WINDOW_WEIGHTS`** — relative influence of each lookback window. Downweight 240m and expanding to emphasise cycle-aware signal over long-run structural anchor.

**`REGIME_IDEALS`** — where each regime sits in (growth, inflation) z-score space. Most impactful calibration lever. Review against actual factor readings from live history.

**`SENTIMENT_RA_THRESHOLD` / `SENTIMENT_BOOST_WEIGHT`** — sensitivity of Sentiment Driven regime to risk appetite signal.

When satisfied with a calibrated set, use "Promote to config.py" in the Config Editor to make it the new default, then delete `config_overrides.json`.

---

## Deployment options

| Option | Effort | Cost | Use case |
|--------|--------|------|----------|
| Local + Task Scheduler | Done | Free | Daily operation, personal use |
| Streamlit Community Cloud | Low | Free | Showcase, demo — no persistent runner |
| Cloud VM (AWS/Azure) | Medium | ~$15/month | Always-on, shared access, proper production |

For always-on deployment without keeping a terminal open: run from a regular Command Prompt window (not VS Code terminal) and minimise rather than close. The server stays live as long as that window exists.

---

## Roadmap

- [x] Four-factor engine — growth, inflation, liquidity, risk appetite
- [x] Five-regime probability vector with confidence scoring
- [x] Multi-window z-score with configurable ensemble weighting
- [x] Historical regime map — asset class performance by regime
- [x] Six-page Streamlit dashboard with live/synthetic toggle
- [x] Config Editor with override system and one-click reset
- [x] Daily runner with state log and change detection
- [x] Live data pipeline — FRED + Yahoo Finance
- [ ] Bloomberg integration — global CB balance sheets, full liquidity factor
- [ ] ALFRED vintage data — epistemically honest bootstrap history
- [ ] Regional expansion — US, Europe, Asia roll-up to global composite
- [ ] Calibration script — score parameter sets against historical anchor points
- [ ] Portfolio construction layer — regime-conditional position sizing
- [ ] Cloud deployment — always-on with authentication

---

## Reference

Fullerton Fund Management, Fullerton Investment Views (FIV), Q1 2023–Q2 2026.
All factor construction and regime taxonomy is inferred from public disclosures.
Fullerton's internal weighting methodology and aggregation rules are proprietary and not replicated here.

---

*This model is a systematic decision-support tool, not investment advice. Past regime-conditional returns are descriptive, not predictive.*
