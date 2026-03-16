# streamlit_app.py
# ============================================================
# Personal Financial View Dashboard
# - Tab 1: FX Monitoring
# - Tab 2: FRED Macro Analysis
# - Tab 3: FRED Detail Charts
# - FX charts: USD/KRW, EUR/KRW
# - Flexible period selection
# - FRED section separated for easier expansion
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
#   FRED_API_KEY="YOUR_FRED_API_KEY"
# ============================================================

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

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
st.caption("FX monitoring + tab-separated FRED macro analysis dashboard")


# ============================================================
# Config
# ============================================================
def get_fred_api_key() -> str:
    key = os.getenv("FRED_API_KEY", "")
    if key:
        return key

    try:
        if "FRED_API_KEY" in st.secrets:
            return st.secrets["FRED_API_KEY"]
    except Exception:
        pass

    return ""


FRED_API_KEY = get_fred_api_key()

FRED_SERIES = {
    "DGS10": {
        "name": "US 10Y Treasury Yield",
        "category": "Rates",
        "description": "Higher long-term yields can pressure equity valuations and tighten financial conditions.",
    },
    "FEDFUNDS": {
        "name": "Fed Funds Rate",
        "category": "Rates",
        "description": "Higher policy rates generally increase financing costs and reduce liquidity.",
    },
    "T10Y2Y": {
        "name": "10Y - 2Y Treasury Spread",
        "category": "Yield Curve",
        "description": "Deep inversion often signals recession risk.",
    },
    "M2SL": {
        "name": "M2 Money Supply",
        "category": "Liquidity",
        "description": "A stronger money/liquidity backdrop can be supportive for risk assets.",
    },
    "CPIAUCSL": {
        "name": "CPI (All Urban Consumers)",
        "category": "Inflation",
        "description": "Higher inflation can pressure bonds and force tighter policy.",
    },
    "UNRATE": {
        "name": "US Unemployment Rate",
        "category": "Labor",
        "description": "Rising unemployment can reflect economic slowdown or recession stress.",
    },
    "BAMLH0A0HYM2": {
        "name": "High Yield OAS",
        "category": "Credit",
        "description": "Wider credit spreads indicate rising financial stress and risk aversion.",
    },
}


# ============================================================
# Helper functions
# ============================================================
def safe_pct_change(series: pd.Series, periods: int = 1) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= periods:
        return None

    prev = s.iloc[-1 - periods]
    curr = s.iloc[-1]

    if prev == 0 or pd.isna(prev) or pd.isna(curr):
        return None

    return (curr / prev - 1.0) * 100.0


def latest_and_prev(series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    s = pd.to_numeric(series, errors="coerce").dropna()
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
    if score <= 4:
        return "Moderate Risk", "🟡"
    return "High Risk", "🔴"


def momentum_label(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if value > 0:
        return "Improving"
    if value < 0:
        return "Deteriorating"
    return "Flat"


def normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index()
    possible_time_cols = ["Date", "Datetime", "date", "datetime", "index"]

    found = None
    for col in possible_time_cols:
        if col in out.columns:
            found = col
            break

    if found is None:
        found = out.columns[0]

    out = out.rename(columns={found: "Time"})
    return out


def ensure_numeric_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ============================================================
# Data loading
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def download_fx_data(period: str = "2y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Yahoo Finance tickers:
    - USD/KRW = KRW=X
    - EUR/KRW = EURKRW=X
    """
    tickers = ["KRW=X", "EURKRW=X"]

    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return {}

    result: Dict[str, pd.DataFrame] = {}

    if isinstance(raw.columns, pd.MultiIndex):
        if "KRW=X" in raw.columns.get_level_values(0):
            fx1 = raw["KRW=X"].copy()
            fx1.columns = [str(c) for c in fx1.columns]
            fx1 = ensure_numeric_column(fx1, "Close")
            result["USDKRW"] = fx1

        if "EURKRW=X" in raw.columns.get_level_values(0):
            fx2 = raw["EURKRW=X"].copy()
            fx2.columns = [str(c) for c in fx2.columns]
            fx2 = ensure_numeric_column(fx2, "Close")
            result["EURKRW"] = fx2

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

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

    if "observations" not in data:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(data["observations"])
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_fred_data() -> Dict[str, pd.DataFrame]:
    return {sid: fetch_fred_series(sid) for sid in FRED_SERIES.keys()}


# ============================================================
# Chart functions
# ============================================================
def make_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_title: str,
    show_ma: bool = True,
) -> go.Figure:
    data = df.copy()

    if x_col not in data.columns:
        fallback_cols = ["Time", "Date", "Datetime", "date", "datetime", "index"]
        found = None
        for col in fallback_cols:
            if col in data.columns:
                found = col
                break
        x_col = found if found is not None else data.columns[0]

    if y_col not in data.columns:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (Missing column: {y_col})", template="plotly_white", height=420)
        return fig

    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode="lines",
            name=y_title,
            line=dict(width=2),
        )
    )

    if show_ma and len(data) >= 50:
        data["MA50"] = data[y_col].rolling(50).mean()
        data["MA200"] = data[y_col].rolling(200).mean()

        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data["MA50"],
                mode="lines",
                name="MA50",
                line=dict(width=1.5, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data["MA200"],
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
# FRED macro logic
# ============================================================
def compute_indicator_risk(series_id: str, df: pd.DataFrame) -> Dict[str, object]:
    meta = FRED_SERIES[series_id]
    s = pd.to_numeric(df["value"], errors="coerce").dropna()

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
    recent_12 = s.iloc[-12:]
    q75 = recent_12.quantile(0.75)

    risk_score = 0

    if series_id == "T10Y2Y":
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
        med = recent_12.median()
        if latest > q75:
            risk_score = 2
        elif latest > med:
            risk_score = 1
        else:
            risk_score = 0
        comment = "Rising unemployment suggests weakening macro conditions."

    elif series_id == "CPIAUCSL":
        yoy = safe_pct_change(s, periods=min(12, len(s) - 1))
        yoy = 0 if yoy is None else yoy
        if yoy > 4.0:
            risk_score = 2
        elif yoy > 2.5:
            risk_score = 1
        else:
            risk_score = 0
        comment = f"Inflation pressure check based on YoY CPI: {format_num(yoy)}%."

    elif series_id == "FEDFUNDS":
        med = recent_12.median()
        if latest > q75:
            risk_score = 2
        elif latest > med:
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
        yoy = safe_pct_change(s, periods=min(12, len(s) - 1))
        yoy = 0 if yoy is None else yoy
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
        "Trend": momentum_label(one_month_change),
        "Risk Score": risk_score,
        "Comment": comment,
    }


def build_macro_scorecard(fred_data: Dict[str, pd.DataFrame]):
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
# Sidebar
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

    fx_period = st.selectbox(
        "FX chart period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
        index=4,
    )

    fx_interval = st.selectbox(
        "FX chart interval",
        options=["1d", "1wk", "1mo"],
        index=0,
    )

    selected_detail_series = st.multiselect(
        "FRED detail indicators",
        options=list(FRED_SERIES.keys()),
        default=["T10Y2Y", "BAMLH0A0HYM2", "UNRATE", "CPIAUCSL"],
        format_func=lambda x: FRED_SERIES[x]["name"],
    )

    show_debug = st.checkbox("Show Debug Info", value=False)


# ============================================================
# Refresh
# ============================================================
if refresh_now:
    st.cache_data.clear()
    st.rerun()

if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()


# ============================================================
# Load data
# ============================================================
fx_data = download_fx_data(period=fx_period, interval=fx_interval)
fred_data = load_all_fred_data() if FRED_API_KEY else {}

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# Render functions
# ============================================================
def render_fx_tab():
    st.subheader("💱 FX Monitoring")

    col1, col2 = st.columns(2)

    with col1:
        if "USDKRW" in fx_data and not fx_data["USDKRW"].empty:
            df = normalize_time_column(fx_data["USDKRW"])
            df = ensure_numeric_column(df, "Close")

            valid = df["Close"].dropna()
            latest = valid.iloc[-1] if len(valid) else np.nan
            prev = valid.iloc[-2] if len(valid) > 1 else np.nan
            delta = latest - prev if pd.notna(latest) and pd.notna(prev) else np.nan

            st.metric("USD / KRW", format_num(latest, 2), format_num(delta, 2) if pd.notna(delta) else "N/A")
            st.plotly_chart(
                make_line_chart(df, "Time", "Close", "USD / KRW Exchange Rate", "KRW per USD", show_ma=True),
                use_container_width=True,
            )
        else:
            st.warning("USD/KRW data not available.")

    with col2:
        if "EURKRW" in fx_data and not fx_data["EURKRW"].empty:
            df = normalize_time_column(fx_data["EURKRW"])
            df = ensure_numeric_column(df, "Close")

            valid = df["Close"].dropna()
            latest = valid.iloc[-1] if len(valid) else np.nan
            prev = valid.iloc[-2] if len(valid) > 1 else np.nan
            delta = latest - prev if pd.notna(latest) and pd.notna(prev) else np.nan

            st.metric("EUR / KRW", format_num(latest, 2), format_num(delta, 2) if pd.notna(delta) else "N/A")
            st.plotly_chart(
                make_line_chart(df, "Time", "Close", "EUR / KRW Exchange Rate", "KRW per EUR", show_ma=True),
                use_container_width=True,
            )
        else:
            st.warning("EUR/KRW data not available.")

    st.info("FX tab is focused on KRW exposure monitoring versus USD and EUR.")

    if show_debug:
        with st.expander("Debug Info"):
            if "USDKRW" in fx_data:
                st.write("USDKRW columns:", normalize_time_column(fx_data["USDKRW"]).columns.tolist())
            if "EURKRW" in fx_data:
                st.write("EURKRW columns:", normalize_time_column(fx_data["EURKRW"]).columns.tolist())


def render_fred_macro_tab():
    st.subheader("🏦 FRED Macro Analysis")

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

    st.markdown("### Macro Scorecard")
    st.dataframe(
        score_df[["Indicator", "Category", "Latest", "1M %Chg", "Trend", "Risk Score", "Comment"]],
        use_container_width=True,
        hide_index=True,
    )

    st.plotly_chart(make_risk_bar_chart(score_df), use_container_width=True)

    with st.expander("Macro Interpretation"):
        st.markdown(
            f"""
**Current regime:** {risk_label}  
**Investment timing view:** {timing_view}

This tab should be the place where your previous **FRED Macro Risk Dashboard** logic is merged.
You can directly extend this section with:
- liquidity block
- rates block
- credit block
- inflation block
- recession warning block
- crisis probability summary
"""
        )


def render_fred_detail_tab():
    st.subheader("📈 FRED Detail Charts")

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

        st.plotly_chart(make_fred_chart(df, meta["name"]), use_container_width=True)


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(
    ["💱 FX Monitoring", "🏦 FRED Macro Analysis", "📈 FRED Detail Charts"]
)

with tab1:
    render_fx_tab()

with tab2:
    render_fred_macro_tab()

with tab3:
    render_fred_detail_tab()


# ============================================================
# Footer
# ============================================================
with st.expander("Suggested Next Updates"):
    st.markdown(
        """
### Recommended upgrades

1. **Import your previous FRED dashboard blocks**
   - Liquidity
   - Rates
   - Credit
   - Inflation
   - Recession warning

2. **Add ETF reaction mapping**
   - TLT / IEF / TIP / DBC / GLD / UUP
   - Link macro regime to ETF behavior

3. **Add signal summary cards**
   - Risk-on
   - Neutral
   - Defensive
   - Inflation hedge priority

4. **Add alert system**
   - USD/KRW threshold
   - EUR/KRW threshold
   - Yield curve inversion
   - High yield spread spike

5. **Add historical regime review**
   - 2008
   - 2020
   - 2022 inflation shock
   - compare current regime to past stress periods

6. **Add portfolio implication section**
   - Equity
   - Bonds
   - Gold
   - Commodities
   - Cash
"""
    )
