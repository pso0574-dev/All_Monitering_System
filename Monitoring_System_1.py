# streamlit_app.py
# ============================================================
# Personal Financial View Dashboard
# - Real-time refresh support
# - View / Hide module toggle
# - Flexible module positioning
# - FX charts: USD/KRW, KRW/EUR
# - FRED-based investment timing / risk analysis
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly yfinance requests
#
# Optional:
#   Set FRED_API_KEY in environment variables
#   or .streamlit/secrets.toml
#   FRED_API_KEY="YOUR_API_KEY"
# ============================================================

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Personal Financial View Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Personal Financial View Dashboard")
st.caption("Real-time FX monitoring + FRED-based macro risk and investment timing dashboard")


# ============================================================
# Config
# ============================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", st.secrets.get("FRED_API_KEY", "")) if hasattr(st, "secrets") else os.getenv("FRED_API_KEY", "")
TODAY = datetime.today().date()

FRED_SERIES = {
    "DGS10": {
        "name": "US 10Y Treasury Yield",
        "category": "Rates",
        "higher_risk": "mixed",
        "description": "Higher long-term yields can pressure equity valuations and tighten financial conditions.",
    },
    "FEDFUNDS": {
        "name": "Fed Funds Rate",
        "category": "Rates",
        "higher_risk": "yes",
        "description": "Higher policy rates generally increase financing costs and reduce liquidity.",
    },
    "T10Y2Y": {
        "name": "10Y - 2Y Treasury Spread",
        "category": "Yield Curve",
        "higher_risk": "no",
        "description": "Deep inversion often signals recession risk.",
    },
    "M2SL": {
        "name": "M2 Money Supply",
        "category": "Liquidity",
        "higher_risk": "no",
        "description": "A stronger money/liquidity backdrop can be supportive for risk assets.",
    },
    "CPIAUCSL": {
        "name": "CPI (All Urban Consumers)",
        "category": "Inflation",
        "higher_risk": "yes",
        "description": "Higher inflation can pressure bonds and force tighter policy.",
    },
    "UNRATE": {
        "name": "US Unemployment Rate",
        "category": "Labor",
        "higher_risk": "yes",
        "description": "Rising unemployment can reflect economic slowdown or recession stress.",
    },
    "BAMLH0A0HYM2": {
        "name": "High Yield OAS",
        "category": "Credit",
        "higher_risk": "yes",
        "description": "Wider credit spreads indicate rising financial stress and risk aversion.",
    },
}


# ============================================================
# Helpers
# ============================================================
def safe_pct_change(series: pd.Series, periods: int = 1) -> Optional[float]:
    if series is None or len(series.dropna()) <= periods:
        return None
    s = series.dropna()
    prev = s.iloc[-1 - periods]
    curr = s.iloc[-1]
    if prev == 0:
        return None
    return (curr / prev - 1.0) * 100.0


def latest_and_prev(series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    s = series.dropna()
    if len(s) < 2:
        return None, None
    return float(s.iloc[-1]), float(s.iloc[-2])


def format_num(x: Optional[float], ndigits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.{ndigits}f}"


def score_to_label(score: int) -> Tuple[str, str]:
    if score <= 2:
        return "Low Risk", "🟢"
    elif score <= 4:
        return "Moderate Risk", "🟡"
    else:
        return "High Risk", "🔴"


def momentum_label(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if value > 0:
        return "Improving"
    elif value < 0:
        return "Deteriorating"
    return "Flat"


# ============================================================
# Data loading
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def download_fx_data(period: str = "2y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Yahoo Finance tickers:
    - USD/KRW = KRW=X
    - EUR/KRW = EURKRW=X  -> KRW/EUR = 1 / (EUR/KRW)
    """
    tickers = ["KRW=X", "EURKRW=X"]
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    result = {}

    # USD/KRW
    if "KRW=X" in raw.columns.get_level_values(0):
        fx1 = raw["KRW=X"].copy()
        fx1 = fx1.rename(columns=str)
        fx1["Close"] = pd.to_numeric(fx1["Close"], errors="coerce")
        result["USDKRW"] = fx1

    # KRW/EUR from EUR/KRW inverse
    if "EURKRW=X" in raw.columns.get_level_values(0):
        fx2 = raw["EURKRW=X"].copy()
        fx2 = fx2.rename(columns=str)
        fx2["Close"] = pd.to_numeric(fx2["Close"], errors="coerce")
        fx2["KRW_per_EUR"] = fx2["Close"]
        fx2["EUR_per_KRW"] = 1.0 / fx2["KRW_per_EUR"]
        result["KRWEUR"] = fx2

    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str, start_date: str = "2010-01-01") -> pd.DataFrame:
    if not FRED_API_KEY:
        return pd.DataFrame(columns=["date", "value"])

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "observations" not in data:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_fred_data() -> Dict[str, pd.DataFrame]:
    return {sid: fetch_fred_series(sid) for sid in FRED_SERIES.keys()}


# ============================================================
# Charts
# ============================================================
def make_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_title: str,
    show_ma: bool = True,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=y_title,
            line=dict(width=2),
        )
    )

    if show_ma and len(df) >= 50:
        df2 = df.copy()
        df2["MA50"] = df2[y_col].rolling(50).mean()
        df2["MA200"] = df2[y_col].rolling(200).mean()

        fig.add_trace(
            go.Scatter(
                x=df2[x_col],
                y=df2["MA50"],
                mode="lines",
                name="MA50",
                line=dict(width=1.5, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df2[x_col],
                y=df2["MA200"],
                mode="lines",
                name="MA200",
                line=dict(width=1.5, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        template="plotly_white",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def make_fred_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["value"],
            mode="lines",
            name=title,
            line=dict(width=2),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        height=360,
        hovermode="x unified",
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def make_risk_bar_chart(score_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=score_df["Indicator"],
            y=score_df["Risk Score"],
            text=score_df["Risk Score"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Macro Risk Score by Indicator",
        xaxis_title="Indicator",
        yaxis_title="Risk Score (0-2)",
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=60),
    )
    return fig


# ============================================================
# Macro risk logic
# ============================================================
def compute_indicator_risk(series_id: str, df: pd.DataFrame) -> Dict[str, object]:
    meta = FRED_SERIES[series_id]
    s = df["value"].dropna()

    if len(s) < 12:
        return {
            "Indicator": meta["name"],
            "Category": meta["category"],
            "Latest": np.nan,
            "1M %Chg": np.nan,
            "Trend": "N/A",
            "Risk Score": 0,
            "Comment": "Not enough data",
        }

    latest = float(s.iloc[-1])
    one_month_change = safe_pct_change(s, periods=min(1, len(s) - 1))

    # rolling stats
    recent_12 = s.iloc[-12:]
    q25 = recent_12.quantile(0.25)
    q75 = recent_12.quantile(0.75)
    trend = momentum_label(one_month_change)

    risk_score = 0

    if series_id == "T10Y2Y":
        # More negative = worse
        if latest < 0:
            risk_score = 2
        elif latest < 0.5:
            risk_score = 1
        else:
            risk_score = 0
        comment = "Yield curve inversion increases recession risk." if latest < 0 else "Yield curve remains positive."
    elif series_id == "BAMLH0A0HYM2":
        if latest > 5.0:
            risk_score = 2
        elif latest > 4.0:
            risk_score = 1
        else:
            risk_score = 0
        comment = "Credit spread widening signals financial stress."
    elif series_id == "UNRATE":
        if latest > q75:
            risk_score = 2
        elif latest > recent_12.median():
            risk_score = 1
        else:
            risk_score = 0
        comment = "Rising unemployment suggests weakening macro conditions."
    elif series_id == "CPIAUCSL":
        yoy = safe_pct_change(s, periods=min(12, len(s)-1))
        if yoy is None:
            yoy = 0
        if yoy > 4.0:
            risk_score = 2
        elif yoy > 2.5:
            risk_score = 1
        else:
            risk_score = 0
        comment = f"Inflation pressure check based on YoY CPI: {format_num(yoy)}%."
    elif series_id == "FEDFUNDS":
        if latest > q75:
            risk_score = 2
        elif latest > recent_12.median():
            risk_score = 1
        else:
            risk_score = 0
        comment = "Higher policy rates reduce liquidity."
    elif series_id == "DGS10":
        if latest > q75:
            risk_score = 1
        else:
            risk_score = 0
        comment = "Higher long yields can compress valuation multiples."
    elif series_id == "M2SL":
        yoy = safe_pct_change(s, periods=min(12, len(s)-1))
        if yoy is None:
            yoy = 0
        if yoy < 0:
            risk_score = 2
        elif yoy < 3.0:
            risk_score = 1
        else:
            risk_score = 0
        comment = f"Liquidity backdrop from M2 YoY: {format_num(yoy)}%."
    else:
        comment = meta["description"]

    return {
        "Indicator": meta["name"],
        "Category": meta["category"],
        "Latest": latest,
        "1M %Chg": one_month_change,
        "Trend": trend,
        "Risk Score": risk_score,
        "Comment": comment,
    }


def build_macro_scorecard(fred_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, int, str, str]:
    rows = []
    total_score = 0

    for sid, df in fred_data.items():
        if df.empty:
            continue
        row = compute_indicator_risk(sid, df)
        rows.append(row)
        total_score += int(row["Risk Score"])

    score_df = pd.DataFrame(rows)

    label, icon = score_to_label(total_score)

    if total_score <= 2:
        timing_view = "Risk-on / constructive backdrop"
    elif total_score <= 4:
        timing_view = "Balanced / selective allocation"
    else:
        timing_view = "Defensive / risk management priority"

    return score_df, total_score, label, f"{icon} {timing_view}"


# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("⚙️ Controls")

    refresh_now = st.button("🔄 Refresh Data", use_container_width=True)

    auto_refresh = st.checkbox("Enable auto refresh", value=False)
    refresh_interval = st.selectbox(
        "Refresh interval",
        options=[30, 60, 120, 300],
        index=1,
        format_func=lambda x: f"{x} sec",
    )

    period = st.selectbox(
        "FX chart period",
        options=["6mo", "1y", "2y", "5y"],
        index=2,
    )

    st.subheader("View / Hide Modules")
    show_fx = st.checkbox("Show FX Monitoring", value=True)
    show_macro = st.checkbox("Show FRED Macro Dashboard", value=True)
    show_detail = st.checkbox("Show Indicator Detail Charts", value=True)

    st.subheader("Module Positioning")
    order_options = {
        "FX Monitoring": 1,
        "Macro Dashboard": 2,
        "Detail Charts": 3,
    }

    pos_fx = st.selectbox("Position - FX Monitoring", [1, 2, 3], index=0)
    pos_macro = st.selectbox("Position - Macro Dashboard", [1, 2, 3], index=1)
    pos_detail = st.selectbox("Position - Detail Charts", [1, 2, 3], index=2)

    st.subheader("FRED Detail Options")
    selected_detail_series = st.multiselect(
        "Indicators to display",
        options=list(FRED_SERIES.keys()),
        default=["T10Y2Y", "BAMLH0A0HYM2", "UNRATE", "CPIAUCSL"],
        format_func=lambda x: FRED_SERIES[x]["name"],
    )

    st.markdown("---")
    st.caption("Tip: duplicate positions are allowed, but unique positions are recommended.")


# ============================================================
# Refresh logic
# ============================================================
if refresh_now:
    st.cache_data.clear()
    st.rerun()

if auto_refresh:
    st.caption(f"⏱️ Auto refresh enabled: every {refresh_interval} seconds")
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()


# ============================================================
# Load data
# ============================================================
fx_data = download_fx_data(period=period)

fred_data = load_all_fred_data() if FRED_API_KEY else {}

last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last updated: {last_update}")


# ============================================================
# Module renderers
# ============================================================
def render_fx_monitoring():
    st.subheader("💱 Exchange Rate Monitoring")

    col1, col2 = st.columns(2)

    # USD/KRW
    with col1:
        if "USDKRW" in fx_data and not fx_data["USDKRW"].empty:
            df = fx_data["USDKRW"].reset_index().rename(columns={"Date": "date"})
            latest = df["Close"].dropna().iloc[-1] if len(df["Close"].dropna()) else np.nan
            prev = df["Close"].dropna().iloc[-2] if len(df["Close"].dropna()) > 1 else np.nan
            delta = latest - prev if pd.notna(latest) and pd.notna(prev) else np.nan

            st.metric(
                "USD / KRW",
                value=format_num(latest, 2),
                delta=format_num(delta, 2) if pd.notna(delta) else "N/A",
            )
            fig = make_line_chart(
                df=df,
                x_col="Date",
                y_col="Close",
                title="USD / KRW Exchange Rate",
                y_title="KRW per USD",
                show_ma=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("USD/KRW data not available.")

    # KRW/EUR
    with col2:
        if "KRWEUR" in fx_data and not fx_data["KRWEUR"].empty:
            df = fx_data["KRWEUR"].reset_index().rename(columns={"Date": "date"})
            latest = df["EUR_per_KRW"].dropna().iloc[-1] if len(df["EUR_per_KRW"].dropna()) else np.nan
            prev = df["EUR_per_KRW"].dropna().iloc[-2] if len(df["EUR_per_KRW"].dropna()) > 1 else np.nan
            delta = latest - prev if pd.notna(latest) and pd.notna(prev) else np.nan

            st.metric(
                "KRW / EUR",
                value=f"{latest:.6f}" if pd.notna(latest) else "N/A",
                delta=f"{delta:.6f}" if pd.notna(delta) else "N/A",
            )
            fig = make_line_chart(
                df=df,
                x_col="Date",
                y_col="EUR_per_KRW",
                title="KRW / EUR Exchange Rate",
                y_title="EUR per KRW",
                show_ma=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("KRW/EUR data not available.")

    st.info(
        "Interpretation: USD/KRW helps monitor KRW weakness vs USD. "
        "KRW/EUR helps track Korean won purchasing power relative to EUR-based expenses/investments."
    )


def render_macro_dashboard():
    st.subheader("🏦 FRED-Based Investment Timing / Risk Analysis")

    if not FRED_API_KEY:
        st.error("FRED_API_KEY not found. Please add your FRED API key.")
        return

    if not fred_data:
        st.warning("No FRED data available.")
        return

    score_df, total_score, risk_label, timing_view = build_macro_scorecard(fred_data)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Macro Risk Score", total_score)
    c2.metric("Risk Regime", risk_label)
    c3.metric("Investment Timing View", timing_view)

    st.dataframe(
        score_df[["Indicator", "Category", "Latest", "1M %Chg", "Trend", "Risk Score", "Comment"]],
        use_container_width=True,
        hide_index=True,
    )

    fig = make_risk_bar_chart(score_df)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Macro Interpretation"):
        st.markdown(
            f"""
**Current regime:** {risk_label}  
**Investment timing view:** {timing_view}

General interpretation:
- **Low Risk:** Macro backdrop is relatively supportive. Risk assets can be favored selectively.
- **Moderate Risk:** Mixed environment. Use balanced allocation and stronger position sizing discipline.
- **High Risk:** Defensive posture preferred. Prioritize quality, duration balance, inflation hedges, and liquidity.
            """
        )


def render_detail_charts():
    st.subheader("📈 FRED Indicator Detail Charts")

    if not FRED_API_KEY:
        st.error("FRED_API_KEY not found. Please add your FRED API key.")
        return

    if not selected_detail_series:
        st.info("Select at least one FRED indicator in the sidebar.")
        return

    for sid in selected_detail_series:
        df = fred_data.get(sid, pd.DataFrame())
        if df.empty:
            st.warning(f"{sid}: data not available.")
            continue

        meta = FRED_SERIES[sid]
        latest, prev = latest_and_prev(df["value"])
        delta = None if latest is None or prev is None else latest - prev

        st.markdown(f"### {meta['name']}")
        c1, c2, c3 = st.columns([1, 1, 3])
        c1.metric("Latest", format_num(latest, 2))
        c2.metric("Change", format_num(delta, 2) if delta is not None else "N/A")
        c3.markdown(f"**Why it matters:** {meta['description']}")

        fig = make_fred_chart(df, meta["name"])
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Dynamic module layout
# ============================================================
modules = []

if show_fx:
    modules.append((pos_fx, "FX Monitoring", render_fx_monitoring))
if show_macro:
    modules.append((pos_macro, "Macro Dashboard", render_macro_dashboard))
if show_detail:
    modules.append((pos_detail, "Detail Charts", render_detail_charts))

modules = sorted(modules, key=lambda x: (x[0], x[1]))

for _, _, render_func in modules:
    render_func()
    st.markdown("---")


# ============================================================
# Footer notes
# ============================================================
with st.expander("Notes / Ideas for Next Upgrade"):
    st.markdown(
        """
- Add **ETF macro watchlist**: TLT / IEF / TIP / DBC / GLD / UUP
- Add **asset allocation suggestion** based on macro regime
- Add **alert thresholds** for USD/KRW, credit spread, and yield curve inversion
- Add **drag-and-drop layout** using a custom component later
- Add **session-state saved layout** per user preference
        """
    )
