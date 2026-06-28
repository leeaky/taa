<context>
# taa — current decisions and rationale

## Status
Live. Full pipeline running on real data. Dashboard deployed locally.

## Purpose
Systematic global macro regime model for Tactical Asset Allocation.
Produces a daily investment environment signal to inform discretionary portfolio decisions.
Modelled on Fullerton Fund Management's four-factor Investment Environment framework.

## What's built
- Four-factor engine: growth, inflation, liquidity, risk appetite — each as z-score deviations from trend across five lookback windows (3yr, 5yr, 10yr, 20yr, full history)
- Five-regime probability vector: Recovery, Goldilocks, Late Cycle, Danger Zone, Sentiment Driven
- Weighted ensemble: window weights configurable — shorter windows emphasise current cycle, longer windows provide structural context
- Three-layer confidence scoring: factor magnitude, secondary confirmation, window consensus
- Historical regime map: asset class performance by regime (BOOTSTRAP label until live state log accumulates)
- Six-page Streamlit dashboard: Current State, Factor History, Regime History, Asset Performance, Data Quality, Config Editor
- Daily runner: fetches data, runs pipeline, appends to state log, exits — scheduled via Windows Task Scheduler
- Config override system: parameter changes saved to config_overrides.json from dashboard, reset without touching code
- Live data pipeline: FRED (via pandas-datareader) + Yahoo Finance

## Key decisions and rationale

**Fullerton as template.** Their four-factor framework is publicly disclosed across FIV quarterly reports Q1 2023–Q2 2026. Factor construction and regime taxonomy inferred from public disclosures. Internal weighting methodology is proprietary and not replicated.

**Five regimes not four.** Fullerton's public model uses five regimes including Sentiment Driven — treated as a distinct positive regime where macro is ambiguous and risk appetite is the primary return driver. Not an uncertainty flag.

**Probabilistic output.** Probability vector across all five regimes rather than a hard label. Consistent with Fullerton's disclosed Investment Environment Indicator which shows stacked probability bars, not single regime calls.

**Multi-window with configurable weights.** Equal weighting of all five windows caused Sentiment Driven dominance because long-run windows anchor to the full-cycle mean. Default: 36m=1.0, 60m=1.0, 120m=1.0, 240m=0.5, expanding=0.5. Adjustable in Config Editor without code changes.

**State log as epistemic record.** Every daily run appends what the model said using only data available at that moment. BOOTSTRAP (revised historical data) and LIVE (state log) periods tagged separately. The model cannot revise what it said in the past.

**Discretionary overlay.** Model produces signal. Portfolio manager decides. Systematic layer handles cognitive bias and data overload; discretionary layer handles model blind spots and genuine turning points.

**Public data first, Bloomberg later.** FRED + Yahoo Finance covers ~70% of what a professional desk would use. Known gaps: liquidity factor is Fed-centric (missing ECB/PBoC/BOJ), OECD CLI runs 2.5yr behind, put/call ratio not on Yahoo Finance. Bloomberg access will replace thin proxies — only fetcher.py and config.py change, nothing else.

**Config-driven throughout.** All parameters in config.py — no hardcoded numbers in logic modules. Calibration = editing config.py or using the dashboard override system.

## Current live readings (as of late June 2026)
- Growth: +0.42 (60m) — above trend
- Inflation: +0.04 (60m) — at trend
- Liquidity: -0.11 (60m) — marginally tight
- Risk Appetite: +0.73 (60m) — clearly positive
- Primary regime: Late Cycle / Sentiment Driven (contested — window divergence)
- Confidence: ~72%
- Transition flag: active

Broadly consistent with Fullerton Q2 2026: "maturing into Late Cycle, Goldilocks rising in probability."

## Known issues / limitations
- OECD CLI data on FRED runs to early 2024 — growth factor uses yield curve, equity momentum, and GDP as compensating sub-components for recent months
- Sentiment Driven historically over-represented on bootstrap data — partly a synthetic data artifact, partly genuine (2014-2019 and post-2023 were legitimately near-trend environments)
- Bootstrap asset performance table not reliable — small samples, short periods, revised data. Do not use for calibration. Use Regime History probability distribution instead.
- State log has 0 live rows — runner needs to be scheduled and accumulate history before live asset performance data is meaningful

## Calibration status
- Window weights set to 36m=1.0, 60m=1.0, 120m=0.8, 240m=0.3, expanding=0.2 (override)
- Softmax temperature at default 1.5 — consider reducing to 1.0 once 3-6 months live data exists
- Regime ideals at defaults — review after live history accumulates
- Recommend: schedule runner daily, accumulate 6 months, then calibrate REGIME_IDEALS against known historical periods (2008=Danger Zone, 2013-2015=Goldilocks, 2022=Late Cycle/Danger Zone)

## File structure
```
C:\dev\taa\
├── src\
│   ├── config.py
│   ├── config_overrides.py
│   ├── runner.py
│   ├── data\
│   ├── factors\
│   ├── regimes\
│   ├── analysis\
│   ├── dashboard\
│   ├── tests\
│   └── logs\              ← created on first runner run
├── .gitignore
├── context.md
├── README.md
└── requirements.txt
```

## Roadmap
- [ ] Schedule runner via Windows Task Scheduler
- [ ] Accumulate 3-6 months live state log
- [ ] Calibrate REGIME_IDEALS against live history
- [ ] Bloomberg integration — global CB balance sheets
- [ ] ALFRED vintage data — epistemically honest bootstrap
- [ ] Regional expansion — US/Europe/Asia roll-up
- [ ] Calibration script — score parameter sets against historical anchors
- [ ] Cloud deployment — always-on with authentication
- [ ] Portfolio construction layer — regime-conditional position sizing
</context>
