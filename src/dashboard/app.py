# =============================================================================
# MODULE 5: STREAMLIT DASHBOARD
# Live presentation layer for the TAA Regime Model.
#
# Run from C:\dev\taa with:
#   streamlit run src/dashboard/app.py
#
# Pages:
#   1. Current State     — factor gauges, regime call, probability vector
#   2. Factor History    — time series with regime shading
#   3. Regime History    — regime timeline, window consensus
#   4. Asset Performance — heatmap of returns by regime × asset
#   5. Data Quality      — staleness, tier, coverage transparency
#   6. Config Editor     — view and edit model parameters, save overrides
# =============================================================================

import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# --- Path setup --------------------------------------------------------------
SRC = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))

# Use config_overrides as drop-in for config — applies any saved overrides
import config_overrides as config
from config_overrides import (
    EDITABLE_PARAMS, load_overrides, save_overrides,
    delete_overrides, get_diff, promote_to_defaults, OVERRIDES_PATH,
    get_effective_config,
)
import config as _base_config
from data.synthetic import generate_synthetic_data
from data.vintage_manager import build_monthly_frame, get_data_quality_summary
from factors.engine import compute_factors, get_factor_summary
from regimes.classifier import classify_regimes, get_current_regime, REGIME_NAMES
from analysis.regime_map import (
    build_regime_map,
    get_current_regime_historical_context,
)

logging.basicConfig(level=logging.WARNING)

# --- Cloud detection ---------------------------------------------------------
# Streamlit Community Cloud sets specific environment variables.
# When running on cloud: use committed data snapshot, disable live toggle.
import os as _os
IS_CLOUD = bool(
    _os.environ.get("STREAMLIT_SHARING_MODE") or
    _os.environ.get("HOSTNAME", "").startswith("streamlit") or
    not Path(SRC / "data" / "cache").exists()  # no cache = likely cloud
)

# =============================================================================
# DESIGN TOKENS
# Dark terminal aesthetic — appropriate for a live macro signal tool.
# Monospace data, sharp lines, colour used only for regime signal.
# =============================================================================

COLORS = {
    "bg":           "#0d1117",
    "surface":      "#161b22",
    "border":       "#21262d",
    "text":         "#e6edf3",
    "text_muted":   "#8b949e",
    "accent":       "#58a6ff",
    "positive":     "#3fb950",
    "negative":     "#f85149",
    "warning":      "#d29922",
}


def hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha) for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

REGIME_COLORS = {r: v["color"] for r, v in config.REGIME_LABELS.items()}
REGIME_COLORS.setdefault("Sentiment Driven", "#9b59b6")

ASSET_LABELS = {
    "^GSPC": "S&P 500",
    "VT":    "Global Eq",
    "EEM":   "EM Eq",
    "TLT":   "US 10y",
    "AGG":   "US Agg",
    "HYG":   "HY Credit",
    "DJP":   "Commod.",
    "GLD":   "Gold",
}

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="TAA Regime Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
    .block-container {{ padding-top: 1.5rem; }}
    [data-testid="stSidebar"] {{ background-color: {COLORS['surface']}; }}
    [data-testid="stMetric"] {{ background-color: {COLORS['surface']};
                                 border: 1px solid {COLORS['border']};
                                 border-radius: 6px; padding: 12px; }}
    h1, h2, h3 {{ color: {COLORS['text']}; font-family: monospace; }}
    .stAlert {{ background-color: {COLORS['surface']}; }}
    div[data-testid="stMetricValue"] {{ font-family: monospace; font-size: 1.4rem; }}
    .regime-badge {{
        display: inline-block; padding: 4px 12px; border-radius: 4px;
        font-family: monospace; font-weight: bold; font-size: 1.1rem;
    }}
    .data-note {{
        font-family: monospace; font-size: 0.75rem;
        color: {COLORS['text_muted']}; padding: 6px 0;
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING — CACHED
# overrides_hash is a stable string key derived from the current overrides
# so that changing parameters busts the cache correctly.
# =============================================================================

def _make_cfg(overrides: dict):
    """
    Build a simple namespace object from base config + overrides.
    Used to pass override-aware config into pipeline functions.
    """
    import types
    cfg = types.SimpleNamespace()
    effective = get_effective_config()
    # Apply any runtime overrides on top
    effective.update(overrides)
    for k, v in effective.items():
        setattr(cfg, k, v)
    return cfg


@st.cache_data(ttl=config.REFRESH_CACHE_HOURS * 3600, show_spinner="Loading data...")
def load_all(use_live: bool, overrides_hash: str):
    """overrides_hash busts cache when parameters change."""
    # Rebuild config fresh from disk — bypasses Streamlit module cache
    overrides = load_overrides()
    cfg = _make_cfg(overrides)

    if use_live:
        try:
            from data.fetcher import get_data
            raw = get_data(cfg)
        except Exception as e:
            st.warning(f"Live data fetch failed ({e}). Falling back to snapshot.")
            from data.fetcher import get_data
            raw = get_data(cfg, use_snapshot=True)
            if not raw:
                from data.synthetic import generate_synthetic_data
                raw, _, _ = generate_synthetic_data()
                use_live = False
    elif IS_CLOUD:
        # Cloud showcase — use committed snapshot of real data
        from data.fetcher import get_data
        raw = get_data(cfg, use_snapshot=True)
        if not raw:
            # Snapshot not committed yet — fall back to synthetic
            st.info("Data snapshot not found — using synthetic data. "
                    "Run src/data/snapshot.py locally and commit the snapshot.")
            from data.synthetic import generate_synthetic_data
            raw, _, _ = generate_synthetic_data()
    else:
        from data.synthetic import generate_synthetic_data
        raw, _, _ = generate_synthetic_data()

    df, quality         = build_monthly_frame(raw, cfg)
    factors_df, meta    = compute_factors(df, cfg)

    # Diagnostic — visible in terminal when running live
    valid_factor_rows = len(factors_df.dropna(how="all"))
    import logging as _log
    _log.getLogger(__name__).warning(
        f"load_all: monthly frame={df.shape}, "
        f"factor rows with data={valid_factor_rows}, "
        f"mode={'LIVE' if use_live else 'SYNTHETIC'}"
    )

    # Determine data mode label for regime map
    if use_live:
        _data_mode = "LIVE"
    elif IS_CLOUD:
        _data_mode = "SNAPSHOT"
    else:
        _data_mode = "BOOTSTRAP"

    ensemble_df, win_df = classify_regimes(factors_df, cfg)
    regime_map          = build_regime_map(ensemble_df, df, cfg,
                                           data_mode=_data_mode)
    factor_summary      = get_factor_summary(factors_df, cfg)
    current             = get_current_regime(ensemble_df)
    quality_df          = get_data_quality_summary(quality)

    return {
        "df":             df,
        "factors_df":     factors_df,
        "ensemble_df":    ensemble_df,
        "window_df":      win_df,
        "regime_map":     regime_map,
        "factor_summary": factor_summary,
        "current":        current,
        "quality_df":     quality_df,
        "use_live":       use_live,
        "loaded_at":      datetime.utcnow(),
    }


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙ Settings")

    use_live = st.toggle(
        "Live data (FRED + Yahoo)",
        value=False,
        disabled=IS_CLOUD,
        help="Live data not available in cloud showcase mode. "
             "Run locally to enable." if IS_CLOUD else
             "Toggle to fetch live data from FRED + Yahoo Finance.",
    )
    if IS_CLOUD:
        st.caption("☁ Cloud mode — using committed data snapshot")

    if st.button("↺  Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["Current State", "Factor History", "Regime History",
         "Asset Performance", "Data Quality", "Config Editor"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div class='data-note'>
    TAA Regime Model<br>
    Inspired by Fullerton Fund Management<br>
    Four-factor · Five-regime<br>
    Public data · Discretionary overlay
    </div>
    """, unsafe_allow_html=True)

# --- Load data ---------------------------------------------------------------
# Compute a stable hash of current overrides to bust cache when params change
import json as _json
_current_overrides = load_overrides()
_overrides_hash = str(hash(_json.dumps(_current_overrides, sort_keys=True)))

data = load_all(use_live, _overrides_hash)

current       = data["current"]
factors_df    = data["factors_df"]
ensemble_df   = data["ensemble_df"]
window_df     = data["window_df"]
regime_map    = data["regime_map"]
factor_summary = data["factor_summary"]
quality_df    = data["quality_df"]

as_of = factors_df.dropna(how="all").index[-1].strftime("%d %b %Y")
if IS_CLOUD and not data["use_live"]:
    data_mode = "SNAPSHOT"
else:
    data_mode = "LIVE" if data["use_live"] else "SYNTHETIC"

# =============================================================================
# HELPERS
# =============================================================================

def regime_badge(regime: str) -> str:
    color = REGIME_COLORS.get(regime, "#666")
    return (f'<span class="regime-badge" '
            f'style="background:{color}22; color:{color}; '
            f'border:1px solid {color}55">{regime}</span>')


def factor_gauge(value: float, label: str) -> go.Figure:
    """Horizontal bar gauge for a single factor score."""
    clamped = max(-3, min(3, value if not np.isnan(value) else 0))
    color   = COLORS["positive"] if clamped > 0 else COLORS["negative"]
    fig = go.Figure(go.Bar(
        x=[clamped], y=[label],
        orientation="h",
        marker_color=color,
        width=0.5,
    ))
    fig.add_vline(x=0, line_color=COLORS["border"], line_width=2)
    fig.update_layout(
        height=70, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-3, 3], showgrid=False,
                   tickvals=[-2, -1, 0, 1, 2],
                   tickfont=dict(color=COLORS["text_muted"], size=9)),
        yaxis=dict(showticklabels=False),
        showlegend=False,
    )
    return fig


def add_regime_shading(fig, ensemble_df, row=1, col=1):
    """
    Add regime background shading using a single scatter fill trace per regime.
    Much faster than add_vrect which creates one shape per period.
    """
    regime_series = ensemble_df["regime_primary"].dropna()
    if regime_series.empty:
        return fig

    # Build one continuous x/y trace per regime using NaN breaks
    # Each regime gets a band at y=[min,max] with gaps where it's not active
    regime_x = {r: [] for r in REGIME_COLORS}
    prev_regime = None

    for dt, regime in regime_series.items():
        if regime not in regime_x:
            regime_x[regime] = []
        if regime != prev_regime:
            # Close previous
            if prev_regime and regime_x[prev_regime]:
                regime_x[prev_regime].append(dt)
                regime_x[prev_regime].append(None)  # break
            # Open new
            regime_x[regime].append(dt)
        prev_regime = regime

    # Close final period
    if prev_regime and regime_x[prev_regime]:
        regime_x[prev_regime].append(regime_series.index[-1])

    for regime, xs in regime_x.items():
        if not xs:
            continue
        color = REGIME_COLORS.get(regime, "#666")
        # Expand x to x0,x1 pairs and build y bands
        expanded_x = []
        expanded_y_top = []
        expanded_y_bot = []
        i = 0
        while i < len(xs):
            if xs[i] is None:
                expanded_x.append(None)
                expanded_y_top.append(None)
                expanded_y_bot.append(None)
                i += 1
            elif i + 1 < len(xs) and xs[i + 1] is not None:
                x0, x1 = xs[i], xs[i + 1]
                expanded_x += [x0, x1, x1, x0, x0, None]
                expanded_y_top += [1, 1, -1, -1, 1, None]
                expanded_y_bot += [1, 1, -1, -1, 1, None]
                i += 2
            else:
                i += 1

        if not expanded_x:
            continue

        fig.add_trace(go.Scatter(
            x=expanded_x,
            y=expanded_y_top,
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.10),
            line=dict(width=0),
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            yaxis=f"y{row if row > 1 else ''}",
        ), row=row, col=col)

    return fig


# =============================================================================
# CACHED CHART BUILDERS
# Charts are expensive to build — cache them separately from data so that
# switching pages or toggling options doesn't rebuild from scratch every time.
# Streamlit passes DataFrames with leading underscore to skip hashing them
# (they're already hashed by load_all's cache).
# =============================================================================

@st.cache_data(ttl=config.REFRESH_CACHE_HOURS * 3600, show_spinner=False)
def build_factor_history_chart(_factors_df, _ensemble_df, window_choice, show_shading):
    """Build the four-panel factor history chart. Cached per window + shading combo."""
    factor_names  = ["growth", "inflation", "liquidity", "risk_appetite"]
    factor_colors = [COLORS["positive"], COLORS["warning"],
                     COLORS["accent"], "#c678dd"]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=["Growth", "Inflation",
                                        "Liquidity", "Risk Appetite"])

    for i, (fname, fcolor) in enumerate(zip(factor_names, factor_colors), 1):
        col = f"{fname}_{window_choice}"
        if col not in _factors_df.columns:
            continue
        s = _factors_df[col].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=fname.replace("_", " ").title(),
            line=dict(color=fcolor, width=1.5),
            fill="tozeroy",
            fillcolor=hex_to_rgba(fcolor, 0.12),
        ), row=i, col=1)
        fig.add_hline(y=0, line_color=COLORS["border"], line_width=1, row=i, col=1)
        if show_shading:
            fig = add_regime_shading(fig, _ensemble_df, row=i, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="monospace", size=11),
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for i in range(1, 5):
        fig.update_yaxes(gridcolor=COLORS["border"],
                         zerolinecolor=COLORS["border"], row=i, col=1)
        fig.update_xaxes(gridcolor=COLORS["border"], row=i, col=1)

    return fig


# =============================================================================
# PAGE ROUTING
# =============================================================================

if page == "Current State":

    # Header
    col_title, col_as_of = st.columns([3, 1])
    with col_title:
        st.markdown("## Current Investment Environment")
    with col_as_of:
        st.markdown(f"""
        <div style='text-align:right; padding-top:10px;
                    color:{COLORS["text_muted"]}; font-family:monospace;
                    font-size:0.85rem;'>
        As of {as_of}<br>{data_mode}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # --- Regime call ---------------------------------------------------------
    regime  = current.get("regime_primary", "—")
    conf    = current.get("regime_confidence", 0) or 0
    trans   = current.get("regime_transition", False)
    desc    = config.REGIME_LABELS.get(regime, {}).get("description", "")

    col_regime, col_conf, col_trans = st.columns([2, 1, 1])

    with col_regime:
        st.markdown("**Primary Regime**")
        st.markdown(regime_badge(regime), unsafe_allow_html=True)
        st.caption(desc)

    with col_conf:
        st.metric("Confidence", f"{conf:.0%}")

    with col_trans:
        flag = "🔄 Transition" if trans else "✓ Stable"
        color = COLORS["warning"] if trans else COLORS["positive"]
        st.markdown(f"""
        <div style='padding-top:8px'>
        <div style='color:{COLORS["text_muted"]}; font-size:0.8rem'>Status</div>
        <div style='color:{color}; font-family:monospace;
                    font-size:1.1rem; font-weight:bold'>{flag}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # --- Probability vector --------------------------------------------------
    st.markdown("**Regime Probabilities**")

    prob_cols = st.columns(len(REGIME_NAMES))
    for i, r in enumerate(REGIME_NAMES):
        key  = f"prob_{r.replace(' ', '_')}"
        prob = current.get(key) or 0
        color = REGIME_COLORS.get(r, "#666")
        with prob_cols[i]:
            st.markdown(f"""
            <div style='text-align:center; padding:8px;
                        border:1px solid {color}44;
                        border-radius:6px; background:{color}11'>
              <div style='color:{color}; font-family:monospace;
                          font-size:1.4rem; font-weight:bold'>
                {prob:.0%}
              </div>
              <div style='color:{COLORS["text_muted"]}; font-size:0.7rem;
                          margin-top:2px'>{r}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # --- Factor gauges -------------------------------------------------------
    st.markdown("**Factor Scores**")

    windows = [f"{w}m" for w in config.ZSCORE_WINDOWS] + (
        ["expanding"] if config.ZSCORE_EXPANDING else []
    )
    ref_window = "60m"

    factor_names = ["growth", "inflation", "liquidity", "risk_appetite"]
    factor_labels = {
        "growth":        "Growth",
        "inflation":     "Inflation",
        "liquidity":     "Liquidity",
        "risk_appetite": "Risk Appetite",
    }

    col_gauges, col_table = st.columns([2, 3])

    with col_gauges:
        for fname in factor_names:
            col = f"{fname}_{ref_window}"
            val = factors_df[col].dropna().iloc[-1] if col in factors_df.columns else np.nan
            label = factor_labels[fname]
            display_val = f"{val:+.2f}" if not np.isnan(val) else "—"
            st.markdown(f"<small style='color:{COLORS['text_muted']}'>"
                        f"{label} ({display_val})</small>", unsafe_allow_html=True)
            st.plotly_chart(
                factor_gauge(val, label),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with col_table:
        st.markdown(f"**Factor scores across all windows**")
        if not factor_summary.empty:
            styled = factor_summary.style.format("{:+.3f}", na_rep="—") \
                .background_gradient(cmap="RdYlGn", vmin=-2, vmax=2,
                                     subset=[c for c in factor_summary.columns
                                             if c != "consensus"]) \
                .format("{:.3f}", subset=["consensus"] if "consensus" in factor_summary.columns else [])
            st.dataframe(styled, use_container_width=True)

    st.markdown("---")

    # --- Per-window consensus ------------------------------------------------
    st.markdown("**Window Consensus**")
    wc1, wc2, wc3, wc4 = st.columns(4)
    for i, fname in enumerate(factor_names):
        col = f"{fname}_consensus"
        val = factors_df[col].dropna().iloc[-1] if col in factors_df.columns else np.nan
        label = factor_labels[fname]
        interp = "High" if val >= 0.9 else "Moderate" if val >= 0.75 else "Low"
        color  = COLORS["positive"] if val >= 0.9 else \
                 COLORS["warning"] if val >= 0.75 else COLORS["negative"]
        with [wc1, wc2, wc3, wc4][i]:
            st.markdown(f"""
            <div style='padding:8px; border:1px solid {COLORS['border']};
                        border-radius:6px; text-align:center'>
              <div style='color:{color}; font-family:monospace;
                          font-size:1.2rem'>{val:.2f}</div>
              <div style='color:{COLORS["text_muted"]}; font-size:0.7rem'>
                {label}<br>{interp}
              </div>
            </div>""", unsafe_allow_html=True)

    # --- Historical context --------------------------------------------------
    st.markdown("---")
    st.markdown("**Historical Performance in This Regime**")
    ctx = get_current_regime_historical_context(regime_map, regime, config)
    if ctx and ctx.get("asset_returns"):
        rows = []
        for asset, stats in ctx["asset_returns"].items():
            rows.append({
                "Asset": ASSET_LABELS.get(asset, asset),
                "Median Ann. Return": stats["median_annualised_return"],
                "Hit Rate (%)": stats["hit_rate"],
                "Median Max DD (%)": stats["median_max_drawdown"],
            })
        ctx_df = pd.DataFrame(rows).set_index("Asset")
        st.dataframe(
            ctx_df.style
                .format({
                    "Median Ann. Return": "{:+.1f}%",
                    "Hit Rate (%)": "{:.0f}%",
                    "Median Max DD (%)": "{:.1f}%",
                })
                .background_gradient(
                    cmap="RdYlGn",
                    subset=["Median Ann. Return"],
                    vmin=-20, vmax=20,
                ),
            use_container_width=True,
        )
        st.caption(regime_map.get("data_note", ""))


# =============================================================================
# PAGE 2: FACTOR HISTORY
# =============================================================================

elif page == "Factor History":
    st.markdown("## Factor History")
    st.caption(f"As of {as_of} · {data_mode}")
    st.markdown("---")

    window_choice = st.select_slider(
        "Z-score window",
        options=[f"{w}m" for w in config.ZSCORE_WINDOWS] + (
            ["expanding"] if config.ZSCORE_EXPANDING else []
        ),
        value="60m",
    )

    show_shading = st.checkbox("Show regime shading", value=True)

    fig = build_factor_history_chart(factors_df, ensemble_df, window_choice, show_shading)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})

    # Window band chart
    st.markdown("---")
    st.markdown("**Factor Score Range Across Windows** (envelope = uncertainty)")
    factor_choice = st.selectbox(
        "Factor", ["growth", "inflation", "liquidity", "risk_appetite"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    windows = [f"{w}m" for w in config.ZSCORE_WINDOWS] + (
        ["expanding"] if config.ZSCORE_EXPANDING else []
    )
    wdf = pd.DataFrame({
        w: factors_df[f"{factor_choice}_{w}"].dropna()
        for w in windows
        if f"{factor_choice}_{w}" in factors_df.columns
    })

    if not wdf.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=wdf.index, y=wdf.max(axis=1),
            fill=None, mode="lines",
            line=dict(color=COLORS["accent"], width=0),
            showlegend=False,
        ))
        fig2.add_trace(go.Scatter(
            x=wdf.index, y=wdf.min(axis=1),
            fill="tonexty",
            fillcolor=hex_to_rgba(COLORS["accent"], 0.15),
            mode="lines",
            line=dict(color=COLORS["accent"], width=0),
            name="Window range",
        ))
        fig2.add_trace(go.Scatter(
            x=wdf.index, y=wdf["60m"] if "60m" in wdf.columns else wdf.iloc[:, 0],
            mode="lines",
            line=dict(color=COLORS["accent"], width=2),
            name="60m (reference)",
        ))
        fig2.add_hline(y=0, line_color=COLORS["border"], line_width=1)
        fig2.update_layout(
            height=300,
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text"], family="monospace", size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.1),
            xaxis=dict(gridcolor=COLORS["border"]),
            yaxis=dict(gridcolor=COLORS["border"]),
        )
        st.plotly_chart(fig2, use_container_width=True,
                        config={"displayModeBar": False})


# =============================================================================
# PAGE 3: REGIME HISTORY
# =============================================================================

elif page == "Regime History":
    st.markdown("## Regime History")
    st.caption(f"As of {as_of} · {data_mode}")
    st.markdown("---")

    show_raw = st.checkbox("Show unsmoothed (raw) regime alongside smoothed", value=False)

    # Stacked probability chart (Fullerton-style)
    st.markdown("**Regime Probability Distribution over Time**")

    prob_cols_map = {
        r: f"prob_{r.replace(' ', '_')}"
        for r in REGIME_NAMES
        if f"prob_{r.replace(' ', '_')}" in ensemble_df.columns
    }

    if prob_cols_map:
        fig = go.Figure()
        for r, col in prob_cols_map.items():
            s = ensemble_df[col].dropna()
            fig.add_trace(go.Bar(
                x=s.index, y=s.values * 100,
                name=r,
                marker_color=REGIME_COLORS.get(r, "#666"),
            ))
        fig.update_layout(
            barmode="stack",
            height=350,
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text"], family="monospace", size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Probability (%)", gridcolor=COLORS["border"],
                       range=[0, 100]),
            xaxis=dict(gridcolor=COLORS["border"]),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    # Per-window regime comparison
    st.markdown("---")
    st.markdown("**Regime Calls by Window** (last 36 months)")

    windows = [f"{w}m" for w in config.ZSCORE_WINDOWS] + (
        ["expanding"] if config.ZSCORE_EXPANDING else []
    )
    regime_win_cols = [f"regime_{w}" for w in windows
                       if f"regime_{w}" in ensemble_df.columns]

    # Short codes for cell display — full name in hover
    REGIME_CODES = {
        "Recovery":         "REC",
        "Goldilocks":       "GLD",
        "Late Cycle":       "LCY",
        "Danger Zone":      "DNG",
        "Sentiment Driven": "SEN",
        None:               "—",
    }

    if regime_win_cols:
        tail = ensemble_df[["regime_primary"] + regime_win_cols].tail(36)
        regime_order = list(config.REGIME_LABELS.keys())
        regime_num   = {r: i for i, r in enumerate(regime_order)}

        heat_data = tail[regime_win_cols].apply(
            lambda col: col.map(lambda x: regime_num.get(str(x), -1) if x else -1)
        )

        # Short code text for cells, full name for hover
        short_text = [
            [REGIME_CODES.get(str(tail[col].iloc[j]), "—") for j in range(len(tail))]
            for col in regime_win_cols
        ]
        hover_text = [
            [str(tail[col].iloc[j]) for j in range(len(tail))]
            for col in regime_win_cols
        ]

        fig = go.Figure(go.Heatmap(
            z=heat_data.values.T,
            x=tail.index,
            y=[c.replace("regime_", "") for c in regime_win_cols],
            colorscale=[
                [i / max(len(regime_order) - 1, 1),
                 REGIME_COLORS.get(r, "#666")]
                for i, r in enumerate(regime_order)
            ],
            showscale=False,
            text=short_text,
            customdata=[[h] for h in hover_text],
            texttemplate="%{text}",
            textfont=dict(size=9, family="monospace"),
            hovertemplate="Window: %{y}<br>Date: %{x}<br>Regime: %{customdata[0]}<extra></extra>",
        ))
        fig.update_layout(
            height=200,
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text"], family="monospace", size=10),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor=COLORS["border"]),
            yaxis=dict(gridcolor=COLORS["border"]),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

        # Legend
        legend_cols = st.columns(len(regime_order))
        for i, r in enumerate(regime_order):
            color = REGIME_COLORS.get(r, "#666")
            code  = REGIME_CODES.get(r, r)
            with legend_cols[i]:
                st.markdown(
                    f"<div style='text-align:center; font-family:monospace;"
                    f"font-size:0.8rem; padding:3px 6px; border-radius:4px;"
                    f"background:{color}22; border:1px solid {color}55;'>"
                    f"<span style='color:{color}; font-weight:bold'>{code}</span>"
                    f"<br><span style='color:{COLORS['text_muted']};"
                    f"font-size:0.7rem'>{r}</span></div>",
                    unsafe_allow_html=True,
                )

    # Confidence over time
    st.markdown("---")
    st.markdown("**Confidence & Transition Flags** (last 36 months)")

    conf_tail = ensemble_df[["regime_confidence", "regime_transition",
                              "confidence_magnitude", "confidence_secondary",
                              "confidence_consensus"]].tail(36)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.7, 0.3])

    for layer, color in [
        ("confidence_magnitude",  COLORS["positive"]),
        ("confidence_secondary",  COLORS["accent"]),
        ("confidence_consensus",  COLORS["warning"]),
        ("regime_confidence",     COLORS["text"]),
    ]:
        fig.add_trace(go.Scatter(
            x=conf_tail.index,
            y=conf_tail[layer],
            name=layer.replace("_", " ").replace("confidence ", "").title(),
            line=dict(color=color, width=1.5 if layer != "regime_confidence" else 2.5),
            opacity=0.8 if layer != "regime_confidence" else 1.0,
        ), row=1, col=1)

    # Transition flags as markers
    transitions = conf_tail[conf_tail["regime_transition"] == True]
    fig.add_trace(go.Scatter(
        x=transitions.index,
        y=[0.5] * len(transitions),
        mode="markers",
        marker=dict(symbol="triangle-up", size=10,
                    color=COLORS["warning"]),
        name="Transition",
    ), row=2, col=1)

    fig.update_layout(
        height=380,
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="monospace", size=11),
        margin=dict(l=10, r=10, t=10, b=60),
        legend=dict(
            orientation="h",
            y=-0.20,
            x=0,
            yanchor="top",
            font=dict(size=10),
        ),
    )
    for r in [1, 2]:
        fig.update_yaxes(gridcolor=COLORS["border"],
                         zerolinecolor=COLORS["border"], row=r, col=1)
        fig.update_xaxes(gridcolor=COLORS["border"], row=r, col=1)

    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})


# =============================================================================
# PAGE 4: ASSET PERFORMANCE
# =============================================================================

elif page == "Asset Performance":
    st.markdown("## Asset Performance by Regime")
    st.caption(f"Historical regime map · {data_mode}")
    st.markdown("---")
    st.caption(regime_map.get("data_note", ""))

    metric = st.radio(
        "Metric",
        ["Median Annualised Return (%)",
         "Hit Rate (% periods positive)",
         "Median Max Drawdown (%)"],
        horizontal=True,
    )

    matrix_key = {
        "Median Annualised Return (%)":   "matrix_return",
        "Hit Rate (% periods positive)":  "matrix_hitrate",
        "Median Max Drawdown (%)":        "matrix_drawdown",
    }[metric]

    mx = regime_map.get(matrix_key, pd.DataFrame())

    if mx.empty:
        st.warning("No data available.")
    else:
        # Rename columns to friendly labels
        mx_display = mx.rename(columns=ASSET_LABELS)

        is_drawdown = "Drawdown" in metric
        cmap = "RdYlGn_r" if is_drawdown else "RdYlGn"
        vmin = -20 if is_drawdown else -20
        vmax = 5   if is_drawdown else  20

        fig = go.Figure(go.Heatmap(
            z=mx_display.values,
            x=list(mx_display.columns),
            y=list(mx_display.index),
            colorscale="RdYlGn_r" if is_drawdown else "RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "—"
                   for v in row]
                  for row in mx_display.values],
            texttemplate="%{text}",
            textfont=dict(size=12, family="monospace"),
            hovertemplate=(
                "Regime: %{y}<br>Asset: %{x}<br>"
                + metric + ": %{text}<extra></extra>"
            ),
            colorbar=dict(
                tickformat=".0f",
                ticksuffix="%",
                thickness=12,
                len=0.8,
            ),
        ))
        fig.update_layout(
            height=320,
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text"], family="monospace", size=12),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    # Detailed table
    st.markdown("---")
    st.markdown("**Full Summary Table**")
    summary = regime_map.get("summary", pd.DataFrame())
    if not summary.empty:
        regime_filter = st.multiselect(
            "Filter by regime",
            options=list(config.REGIME_LABELS.keys()),
            default=list(config.REGIME_LABELS.keys()),
        )
        filtered = summary[summary["regime"].isin(regime_filter)].copy()
        filtered["asset"] = filtered["asset"].map(
            lambda x: ASSET_LABELS.get(x, x)
        )
        st.dataframe(
            filtered.style.format({
                "median_total_return":      "{:+.1f}%",
                "mean_total_return":        "{:+.1f}%",
                "median_annualised_return": "{:+.1f}%",
                "hit_rate":                 "{:.0f}%",
                "median_max_drawdown":      "{:.1f}%",
                "avg_duration_months":      "{:.1f}m",
            }).background_gradient(
                cmap="RdYlGn", subset=["median_annualised_return"],
                vmin=-20, vmax=20,
            ),
            use_container_width=True,
        )


# =============================================================================
# PAGE 5: DATA QUALITY
# =============================================================================

elif page == "Data Quality":
    st.markdown("## Data Quality")
    st.caption(f"As of {as_of} · {data_mode}")
    st.markdown("---")

    if not quality_df.empty:
        # Traffic light display
        st.markdown("**Series Status**")

        stale_series   = quality_df[quality_df["stale"] == True]
        current_series = quality_df[quality_df["stale"] == False]

        col_ok, col_stale = st.columns(2)

        with col_ok:
            st.markdown(f"✓ **{len(current_series)} series current**")
            for sid, row in current_series.iterrows():
                tier_color = (COLORS["positive"] if row["tier"] == "REALTIME"
                              else COLORS["warning"])
                st.markdown(
                    f"<span style='font-family:monospace; font-size:0.85rem'>"
                    f"<span style='color:{tier_color}'>{row['tier'][:1]}</span> "
                    f"{sid}  "
                    f"<span style='color:{COLORS['text_muted']}'>"
                    f"{row['last_period']}</span></span>",
                    unsafe_allow_html=True,
                )

        with col_stale:
            if stale_series.empty:
                st.markdown("✓ No stale series")
            else:
                st.markdown(f"⚠ **{len(stale_series)} series stale**")
                for sid, row in stale_series.iterrows():
                    neg = COLORS["negative"]
                    st.markdown(
                        f"<span style='font-family:monospace; font-size:0.85rem;"
                        f"color:{neg}'>"
                        f"⚠ {sid}  {row['last_period']}  — {row['reason']}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.markdown("**Data Tier Legend**")
        st.markdown(f"""
        <div style='font-family:monospace; font-size:0.85rem;
                    color:{COLORS["text_muted"]}'>
        <span style='color:{COLORS["positive"]}'>R</span> REALTIME
        — market prices, no revision risk<br>
        <span style='color:{COLORS["warning"]}'>V</span> REVISED
        — standard FRED data, retrospectively clean but not real-time vintage<br>
        <br>
        REVISED series are used as-is for the historical bootstrap.
        They are clearly labelled as illustrative.<br>
        Bloomberg DDIS or Haver Analytics would provide true real-time vintage data.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Full Quality Table**")
        st.dataframe(quality_df, use_container_width=True)

    else:
        st.info("No data quality information available.")

    st.markdown("---")
    st.markdown("**Model Run Info**")
    st.markdown(f"""
    <div style='font-family:monospace; font-size:0.85rem;
                color:{COLORS["text_muted"]}'>
    Data mode:        {data_mode}<br>
    Dashboard loaded: {data['loaded_at'].strftime('%Y-%m-%d %H:%M UTC')}<br>
    Cache TTL:        {config.REFRESH_CACHE_HOURS}h<br>
    Z-score windows:  {config.ZSCORE_WINDOWS} + expanding={config.ZSCORE_EXPANDING}<br>
    Softmax temp:     {config.SOFTMAX_TEMPERATURE}<br>
    Smoothing window: {config.SMOOTHING_WINDOW} months
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 6: CONFIG EDITOR
# =============================================================================

elif page == "Config Editor":
    st.markdown("## Config Editor")
    st.caption("Edit model parameters · Changes saved to config_overrides.json · "
               "Does not modify config.py")
    st.markdown("---")

    # Load current overrides
    overrides = load_overrides()
    diff      = get_diff(overrides)

    # --- Status banner -------------------------------------------------------
    if overrides:
        override_file = str(OVERRIDES_PATH.name)
        st.markdown(
            f"<div style='padding:10px; border-radius:6px; "
            f"border:1px solid {COLORS['warning']}44; "
            f"background:{COLORS['warning']}11; "
            f"font-family:monospace; font-size:0.85rem; "
            f"color:{COLORS['warning']}'>"
            f"⚠  Active overrides in {override_file} — "
            f"{len(diff)} parameter(s) differ from defaults"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='padding:10px; border-radius:6px; "
            f"border:1px solid {COLORS['positive']}44; "
            f"background:{COLORS['positive']}11; "
            f"font-family:monospace; font-size:0.85rem; "
            f"color:{COLORS['positive']}'>"
            f"✓  Running on defaults — no overrides active"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Parameter editor by group -------------------------------------------
    groups = {}
    for key, meta in EDITABLE_PARAMS.items():
        g = meta["group"]
        groups.setdefault(g, []).append((key, meta))

    new_overrides = dict(overrides)  # start from current overrides

    for group_name, params in groups.items():
        st.markdown(f"**{group_name}**")

        for key, meta in params:
            current_val = overrides.get(key, getattr(config._base, key, None))
            default_val = getattr(config._base, key, None)
            is_overridden = key in diff

            label = meta["label"]
            if is_overridden:
                label = f"🔶 {label}"  # highlight overridden params

            ptype = meta["type"]

            if ptype == "float":
                new_val = st.slider(
                    label,
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=float(current_val),
                    step=float(meta.get("step", 0.05)),
                    help=f"{meta['help']}  Default: {default_val}",
                    key=f"slider_{key}",
                )
                if new_val != default_val:
                    new_overrides[key] = new_val
                elif key in new_overrides:
                    del new_overrides[key]

            elif ptype == "int":
                new_val = st.slider(
                    label,
                    min_value=int(meta["min"]),
                    max_value=int(meta["max"]),
                    value=int(current_val),
                    step=1,
                    help=f"{meta['help']}  Default: {default_val}",
                    key=f"slider_{key}",
                )
                if new_val != default_val:
                    new_overrides[key] = new_val
                elif key in new_overrides:
                    del new_overrides[key]

            elif ptype == "regime_ideals":
                st.markdown(
                    f"<small style='color:{COLORS['text_muted']}'>"
                    f"{meta['help']}</small>",
                    unsafe_allow_html=True,
                )
                ideal_overrides = overrides.get("REGIME_IDEALS",
                                                dict(config._base.REGIME_IDEALS))
                ideal_new = {}
                regime_names = list(config._base.REGIME_IDEALS.keys())
                cols = st.columns(len(regime_names))
                for i, rname in enumerate(regime_names):
                    with cols[i]:
                        color = config._base.REGIME_LABELS.get(rname, {}).get("color", "#666")
                        st.markdown(
                            f"<div style='color:{color}; font-family:monospace; "
                            f"font-size:0.8rem; font-weight:bold'>{rname}</div>",
                            unsafe_allow_html=True,
                        )
                        default_g, default_i = config._base.REGIME_IDEALS[rname]
                        cur_g, cur_i = ideal_overrides.get(rname, (default_g, default_i))
                        new_g = st.number_input(
                            "Growth z", value=float(cur_g), step=0.1,
                            min_value=-3.0, max_value=3.0,
                            key=f"ideal_g_{rname}",
                        )
                        new_i = st.number_input(
                            "Inflation z", value=float(cur_i), step=0.1,
                            min_value=-3.0, max_value=3.0,
                            key=f"ideal_i_{rname}",
                        )
                        ideal_new[rname] = (new_g, new_i)

                if ideal_new != dict(config._base.REGIME_IDEALS):
                    new_overrides["REGIME_IDEALS"] = ideal_new
                elif "REGIME_IDEALS" in new_overrides:
                    del new_overrides["REGIME_IDEALS"]

            elif ptype == "regime_secondary":
                st.markdown(
                    f"<small style='color:{COLORS['text_muted']}'>"
                    f"{meta['help']}</small>",
                    unsafe_allow_html=True,
                )
                sec_overrides = overrides.get("REGIME_SECONDARY_IDEALS",
                                              dict(config._base.REGIME_SECONDARY_IDEALS))
                sec_new = {}
                regime_names = list(config._base.REGIME_SECONDARY_IDEALS.keys())
                cols = st.columns(len(regime_names))
                for i, rname in enumerate(regime_names):
                    with cols[i]:
                        color = config._base.REGIME_LABELS.get(rname, {}).get("color", "#666")
                        st.markdown(
                            f"<div style='color:{color}; font-family:monospace;"
                            f"font-size:0.8rem; font-weight:bold'>{rname}</div>",
                            unsafe_allow_html=True,
                        )
                        default_l, default_ra = config._base.REGIME_SECONDARY_IDEALS[rname]
                        cur_l, cur_ra = sec_overrides.get(rname, (default_l, default_ra))
                        new_l = st.number_input(
                            "Liquidity z", value=float(cur_l), step=0.1,
                            min_value=-3.0, max_value=3.0,
                            key=f"sec_l_{rname}",
                        )
                        new_ra = st.number_input(
                            "Risk App z", value=float(cur_ra), step=0.1,
                            min_value=-3.0, max_value=3.0,
                            key=f"sec_ra_{rname}",
                        )
                        sec_new[rname] = (new_l, new_ra)

                if sec_new != dict(config._base.REGIME_SECONDARY_IDEALS):
                    new_overrides["REGIME_SECONDARY_IDEALS"] = sec_new
                elif "REGIME_SECONDARY_IDEALS" in new_overrides:
                    del new_overrides["REGIME_SECONDARY_IDEALS"]

            elif ptype == "window_weights":
                st.markdown(
                    f"<small style='color:{COLORS['text_muted']}'>"
                    f"{meta['help']}</small>",
                    unsafe_allow_html=True,
                )
                default_weights = dict(config._base.ZSCORE_WINDOW_WEIGHTS)
                cur_weights = overrides.get("ZSCORE_WINDOW_WEIGHTS", default_weights)
                ww_new = {}

                all_windows = (
                    [f"{w}m" for w in config._base.ZSCORE_WINDOWS]
                    + (["expanding"] if config._base.ZSCORE_EXPANDING else [])
                )
                cols = st.columns(len(all_windows))
                for i, w in enumerate(all_windows):
                    with cols[i]:
                        default_w = default_weights.get(w, 1.0)
                        cur_w = cur_weights.get(w, default_w)
                        new_w = st.number_input(
                            w,
                            value=float(cur_w),
                            step=0.1,
                            min_value=0.0,
                            max_value=2.0,
                            key=f"ww_{w}",
                            help=f"Default: {default_w}",
                        )
                        ww_new[w] = new_w

                if ww_new != default_weights:
                    new_overrides["ZSCORE_WINDOW_WEIGHTS"] = ww_new
                elif "ZSCORE_WINDOW_WEIGHTS" in new_overrides:
                    del new_overrides["ZSCORE_WINDOW_WEIGHTS"]

        st.markdown("")  # spacing between groups

    # --- Action buttons ------------------------------------------------------
    st.markdown("---")
    st.markdown("**Actions**")

    col_save, col_reset, col_promote = st.columns(3)

    with col_save:
        if st.button("💾  Save overrides", use_container_width=True,
                     type="primary"):
            save_overrides(new_overrides)
            st.cache_data.clear()
            st.success("Overrides saved. Model will reload with new parameters.")
            st.rerun()

    with col_reset:
        if st.button("↺  Reset to defaults", use_container_width=True):
            delete_overrides()
            st.cache_data.clear()
            # Clear slider session state so widgets redraw at default values
            keys_to_clear = [k for k in st.session_state
                             if k.startswith(("slider_", "ideal_", "sec_", "ww_"))]
            for k in keys_to_clear:
                del st.session_state[k]
            st.success("Overrides deleted. Running on defaults.")
            st.rerun()

    with col_promote:
        if st.button("⬆  Promote to config.py", use_container_width=True):
            if not overrides:
                st.warning("No overrides to promote.")
            else:
                result = promote_to_defaults(overrides)
                st.success(f"Written to config.py:\n{result}")
                st.info("Dict parameters (REGIME_IDEALS etc) must be "
                        "updated manually in config.py. "
                        "Delete config_overrides.json when done.")

    # --- Current diff view ---------------------------------------------------
    if diff:
        st.markdown("---")
        st.markdown("**Active overrides vs defaults**")
        diff_rows = []
        for key, vals in diff.items():
            diff_rows.append({
                "Parameter": key,
                "Default": str(vals["default"]),
                "Override": str(vals["override"]),
            })
        diff_df = pd.DataFrame(diff_rows)
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

    # --- Raw JSON view -------------------------------------------------------
    with st.expander("Raw overrides JSON"):
        st.json(new_overrides if new_overrides else {})

