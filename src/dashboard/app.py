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

import config
from data.synthetic import generate_synthetic_data
from data.vintage_manager import build_monthly_frame, get_data_quality_summary
from factors.engine import compute_factors, get_factor_summary
from regimes.classifier import classify_regimes, get_current_regime, REGIME_NAMES
from analysis.regime_map import (
    build_regime_map,
    get_current_regime_historical_context,
)

logging.basicConfig(level=logging.WARNING)

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
# =============================================================================

@st.cache_data(ttl=config.REFRESH_CACHE_HOURS * 3600, show_spinner="Loading data...")
def load_all(use_live: bool):
    if use_live:
        try:
            from data.fetcher import get_data
            raw = get_data(config)
        except Exception as e:
            st.warning(f"Live data fetch failed ({e}). Falling back to synthetic.")
            raw, _, _ = generate_synthetic_data()
            use_live = False
    else:
        raw, _, _ = generate_synthetic_data()

    df, quality         = build_monthly_frame(raw, config)
    factors_df, meta    = compute_factors(df, config)
    ensemble_df, win_df = classify_regimes(factors_df, config)
    regime_map          = build_regime_map(ensemble_df, df, config)
    factor_summary      = get_factor_summary(factors_df, config)
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
        help="Off = synthetic data. On = live FRED + Yahoo feeds.",
    )

    if st.button("↺  Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["Current State", "Factor History", "Regime History",
         "Asset Performance", "Data Quality"],
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
data = load_all(use_live)

current       = data["current"]
factors_df    = data["factors_df"]
ensemble_df   = data["ensemble_df"]
window_df     = data["window_df"]
regime_map    = data["regime_map"]
factor_summary = data["factor_summary"]
quality_df    = data["quality_df"]

as_of = factors_df.dropna(how="all").index[-1].strftime("%d %b %Y")
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
    """Add coloured regime background bands to a plotly figure."""
    prev_regime = None
    prev_date   = None
    for dt, regime in ensemble_df["regime_primary"].dropna().items():
        if regime != prev_regime:
            if prev_regime is not None:
                fig.add_vrect(
                    x0=prev_date, x1=dt,
                    fillcolor=REGIME_COLORS.get(prev_regime, "#444"),
                    opacity=0.08, layer="below", line_width=0,
                    row=row, col=col,
                )
            prev_regime = regime
            prev_date   = dt
    if prev_regime and prev_date:
        fig.add_vrect(
            x0=prev_date, x1=ensemble_df.index[-1],
            fillcolor=REGIME_COLORS.get(prev_regime, "#444"),
            opacity=0.08, layer="below", line_width=0,
            row=row, col=col,
        )
    return fig


# =============================================================================
# PAGE 1: CURRENT STATE
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

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=[
                            "Growth", "Inflation",
                            "Liquidity", "Risk Appetite"
                        ])

    factor_names = ["growth", "inflation", "liquidity", "risk_appetite"]
    factor_colors = [COLORS["positive"], COLORS["warning"],
                     COLORS["accent"], "#c678dd"]

    for i, (fname, fcolor) in enumerate(zip(factor_names, factor_colors), 1):
        col = f"{fname}_{window_choice}"
        if col not in factors_df.columns:
            continue
        s = factors_df[col].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=fname.replace("_", " ").title(),
            line=dict(color=fcolor, width=1.5),
            fill="tozeroy",
            fillcolor=f"{fcolor}15",
        ), row=i, col=1)
        fig.add_hline(y=0, line_color=COLORS["border"],
                      line_width=1, row=i, col=1)

        if show_shading:
            fig = add_regime_shading(fig, ensemble_df, row=i, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="monospace", size=11),
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for i in range(1, 5):
        fig.update_yaxes(
            gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
            row=i, col=1,
        )
        fig.update_xaxes(gridcolor=COLORS["border"], row=i, col=1)

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
            fillcolor=f"{COLORS['accent']}20",
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

    if regime_win_cols:
        tail = ensemble_df[["regime_primary"] + regime_win_cols].tail(36)
        # Map to numeric for heatmap display
        regime_order = list(config.REGIME_LABELS.keys())
        regime_num   = {r: i for i, r in enumerate(regime_order)}

        heat_data = tail[regime_win_cols].applymap(
            lambda x: regime_num.get(str(x), -1) if x else -1
        )

        fig = go.Figure(go.Heatmap(
            z=heat_data.values.T,
            x=tail.index,
            y=[c.replace("regime_", "") for c in regime_win_cols],
            colorscale=[
                [i / (len(regime_order) - 1),
                 REGIME_COLORS.get(r, "#666")]
                for i, r in enumerate(regime_order)
            ],
            showscale=False,
            text=[[str(tail[col].iloc[j]) for j in range(len(tail))]
                  for col in regime_win_cols],
            texttemplate="%{text}",
            textfont=dict(size=8),
            hovertemplate="Window: %{y}<br>Date: %{x}<br>Regime: %{text}<extra></extra>",
        ))
        fig.update_layout(
            height=220,
            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text"], family="monospace", size=10),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor=COLORS["border"]),
            yaxis=dict(gridcolor=COLORS["border"]),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

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
        height=350,
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="monospace", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.05),
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
