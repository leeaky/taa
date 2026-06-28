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

**Confidence scoring — three layers:**
1. Factor magnitude — how far are Growth and Inflation from zero
2. Secondary confirmation — do Liquidity and Risk Appetite support the regime call
3. Window consensus — what fraction of the five window variants agree

---

## Architecture

```
src/
├── config.py               # All parameters — tickers, windows, thresholds, lags
├── data/
│   ├── fetcher.py          # Pull from FRED + Yahoo Finance, cache locally
│   ├── vintage_manager.py  # Lag offsets, staleness flagging, revision signals
│   ├── state_log.py        # Append-only record of every model run
│   └── synthetic.py        # Regime-aware synthetic data for testing/showcase
├── factors/
│   └── engine.py           # Four-factor computation across all z-score windows
├── regimes/
│   └── classifier.py       # Five-regime probability vector + confidence scoring
├── analysis/
│   └── regime_map.py       # Historical asset class performance by regime
└── dashboard/
    └── app.py              # Streamlit dashboard — live presentation layer
```

---

## Key design decisions

**Data vintaging.** The model only uses data that was publicly available at the time of each run. Publication lags are encoded per series in config. Staleness is flagged visibly, never silently filled. Revisions are treated as new information, not corrections to history.

**Window parameterisation.** Every factor is computed across five lookback windows simultaneously. A 3-year window asks "how does today compare to recent history." A 20-year window asks "how does today compare to a full cycle." These are different questions and will sometimes give different answers — that disagreement is surfaced, not hidden.

**State log.** Every daily run appends one row recording what the model said and what data it used. This is the live epistemic record. The historical bootstrap (before day one of live running) uses revised public data and is clearly labelled as illustrative.

**Daily cadence.** The model runs daily. Most days nothing changes. On days when a key release drops (PMI, CPI, GDP), the model updates immediately and logs what changed and why.

**Discretionary overlay.** The model produces a signal. A human decides what to do with it. This is by design — the systematic layer compensates for cognitive bias and data overload; the discretionary layer compensates for model blind spots and regime transitions.

---

## Data sources

| Source | Series | Tier |
|--------|--------|------|
| FRED | GDP, CPI, PCE, M2, Fed balance sheet, yield curve, TIPS, forward inflation | REVISED* |
| Yahoo Finance | VIX, put/call ratio, S&P 500, global equity ETFs, credit ETFs, commodities, gold | REALTIME |
| OECD (via FRED) | Composite Leading Indicator | REVISED* |

*REVISED means standard FRED data — retrospectively clean but not what existed in real time. Flagged in dashboard. ALFRED vintage data and Bloomberg DDIS would close this gap.

**Bloomberg gap:** The liquidity factor is the weakest on public data. ECB, PBoC, and BOJ balance sheet data, along with global M2 composites, require Bloomberg or CEIC. The current proxy (Fed-centric) is flagged and will be replaced when Bloomberg access is available.

---

## Build status

| Module | Status | Notes |
|--------|--------|-------|
| `config.py` | ✅ Complete | Five-regime taxonomy, all tickers and lags |
| `data/fetcher.py` | ✅ Complete | FRED + Yahoo, incremental refresh, cache |
| `data/vintage_manager.py` | ✅ Complete | Lag offsets, staleness flagging, revision signals |
| `data/state_log.py` | ✅ Complete | Append-only, append-on-run |
| `data/synthetic.py` | ✅ Complete | Regime-aware synthetic data, full schema match |
| `factors/engine.py` | ✅ Complete | All four factors, five windows, consensus scoring |
| `regimes/classifier.py` | 🔲 Next | Five-regime probability vector + confidence |
| `analysis/regime_map.py` | 🔲 Pending | Historical asset performance by regime |
| `dashboard/app.py` | 🔲 Pending | Streamlit — live dashboard |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run with synthetic data (no API keys needed)
cd src
python -c "
import config
from data.synthetic import generate_synthetic_data
from data.vintage_manager import build_monthly_frame
from factors.engine import compute_factors, get_factor_summary

raw, _, _ = generate_synthetic_data()
df, quality = build_monthly_frame(raw, config)
factors_df, meta = compute_factors(df, config)
print(get_factor_summary(factors_df, config))
"

# Run with live data (requires internet)
python -c "
import config
from data.fetcher import get_data
from data.vintage_manager import build_monthly_frame
from factors.engine import compute_factors, get_factor_summary

raw = get_data(config)
df, quality = build_monthly_frame(raw, config)
factors_df, meta = compute_factors(df, config)
print(get_factor_summary(factors_df, config))
"

# Launch dashboard (once complete)
streamlit run src/dashboard/app.py
```

---

## Roadmap

- [ ] Module 3: Regime classifier — five-regime probability vector
- [ ] Module 4: Historical regime map — asset class performance by regime
- [ ] Module 5: Streamlit dashboard — live presentation layer
- [ ] Bloomberg integration — replace thin liquidity proxies
- [ ] Regional expansion — US, Europe, Asia roll-up to global
- [ ] ALFRED vintage data — close the real-time data gap for bootstrap history
- [ ] Augmented intelligence layer — alternative data, NLP sentiment signals

---

## Reference

Fullerton Fund Management, Fullerton Investment Views (FIV), Q1 2023–Q2 2026.
All factor construction and regime taxonomy is inferred from public disclosures.
Fullerton's internal weighting methodology and aggregation rules are proprietary and not replicated here.

---

*This model is a systematic decision-support tool, not investment advice. Past regime-conditional returns are descriptive, not predictive.*
