# streamlit_app.py
# ============================================================
# Personal Financial View Dashboard
# - Tab 1: FX & Macro Market Monitoring
# - Tab 2: FRED Macro Analysis
# - Tab 3: FRED Detail Charts
#
# FX & Macro Market Monitoring:
# - Summary table first
# - Charts shown below
# - USD/KRW, EUR/KRW, EUR/USD, DXY
# - BTC/EUR, WTI, Brent, Gold, Silver, Natural Gas
#
# Features:
# - Flexible period / interval selection
# - Real-time refresh support
# - Moving averages
# - 52-week range position
# - Z-score
# - Compact summary table
#
# Run:
#   streamlit run streamlit_app.py
#
# Install:
#   pip install streamlit pandas numpy plotly yfinance requests
#
# Optional:
#   .streamlit/secrets.toml
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
st.caption("FX monitoring + macro market assets + FRED macro analysis dashboard")


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

MARKET_TICKERS = {
    "USDKRW": {
        "ticker": "KRW=X",
        "label": "USD / KRW",
        "y_title": "KRW per USD",
        "interpretation": "Higher USD/KRW usually means KRW weakness and stronger USD pressure.",
    },
    "EURKRW": {
        "ticker": "EURKRW=X",
        "label": "EUR / KRW",
        "y_title": "KRW per EUR",
        "interpretation": "Higher EUR/KRW can increase Germany-based living and spending burden in KRW terms.",
    },
    "EURUSD": {
        "ticker": "EURUSD=X",
        "label": "EUR / USD",
        "y_title": "USD per EUR",
        "interpretation": "EUR/USD helps track broad EUR versus USD strength.",
    },
    "DXY": {
        "ticker": "DX-Y.NYB",
        "label": "US Dollar Index (DXY)",
        "y_title": "Index",
        "interpretation": "A stronger DXY often tightens global financial conditions.",
    },
    "BTCEUR": {
        "ticker": "BTC-EUR",
        "label": "BTC / EUR",
        "y_title": "EUR per BTC",
        "interpretation": "BTC/EUR can reflect liquidity and risk appetite.",
    },
    "WTI": {
        "ticker": "CL=F",
        "label": "WTI Oil",
        "y_title": "USD",
        "interpretation": "Rising oil can signal inflation and cost pressure.",
    },
    "BRENT": {
        "ticker": "BZ=F",
        "label": "Brent Oil",
        "y_title": "USD",
        "interpretation": "Brent is a key global oil benchmark and inflation-sensitive asset.",
    },
    "GOLD": {
        "ticker": "GC=F",
        "label": "Gold",
        "y_title": "USD",
        "interpretation": "Gold often benefits from uncertainty, lower real yields, or hedge demand.",
    },
    "SILVER": {
        "ticker": "SI=F",
        "label": "Silver",
        "y_title": "USD",
        "interpretation": "Silver has both inflation/precious-metal and industrial demand characteristics.",
    },
    "NATGAS": {
        "ticker": "NG=F",
        "label": "Natural Gas",
        "y_title": "USD",
        "interpretation": "Natural gas is important for energy-cost and industrial demand monitoring.",
    },
}

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


def format_pct(x: Optional[float], ndigits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.{ndigits}f}%"


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


def calc_zscore(series: pd.Series, window: int = 60) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < max(window, 20):
        return None

    recent = s.iloc[-window:]
    mean_val = recent.mean()
    std_val = recent.std()

    if std_val == 0 or pd.isna(std_val):
        return None

    return float((recent.iloc[-1] - mean_val) / std_val)


def calc_range_position(series: pd.Series, window: int = 252) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None

    window = min(window, len(s))
    recent = s.iloc[-window:]
    low_val = recent.min()
    high_val = recent.max()
    last_val = recent.iloc[-1]

    if high_val == low_val:
        return None

    return float((last_val - low_val) / (high_val - low_val) * 100.0)


def calc_drawdown_from_high(series: pd.Series, window: int = 252) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None

    window = min(window, len(s))
    recent = s.iloc[-window:]
    high_val = recent.max()
    last_val = recent.iloc[-1]

    if high_val == 0:
        return None

    return float((last_val / high_val - 1.0) * 100.0)


def get_period_days(period: str) -> int:
    mapping = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
        "max": 5000,
    }
    return mapping.get(period, 365)


# ============================================================
# Data loading
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def download_market_data(period: str = "2y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    tickers = [v["ticker"] for v in MARKET_TICKERS.values()]

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
        level0 = raw.columns.get_level_values(0)

        for key, meta in MARKET_TICKERS.items():
            tk = meta["ticker"]
            if tk in level0:
                df = raw[tk].copy()
                df.columns = [str(c) for c in df.columns]
                if "Close" in df.columns:
                    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                result[key] = df

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
        fig.update_layout(
            title=f"{title} (Missing column: {y_col})",
            template="plotly_white",
            height=380,
        )
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
        height=380,
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
# Market summary + chart helpers
# ============================================================
def build_market_summary_table(data_dict: Dict[str, pd.DataFrame], keys: list[str], period_label: str) -> pd.DataFrame:
    rows = []
    period_days = get_period_days(period_label)

    for key in keys:
        meta = MARKET_TICKERS[key]

        if key not in data_dict or data_dict[key].empty:
            rows.append({
                "Asset": meta["label"],
                "Latest": np.nan,
                "1-Step %": np.nan,
                "Z-score(60)": np.nan,
                "Range Pos(52W)": np.nan,
                "From High": np.nan,
                "Interpretation": "Data not available",
            })
            continue

        df = normalize_time_column(data_dict[key])
        df = ensure_numeric_column(df, "Close")
        valid = df["Close"].dropna()

        if len(valid) == 0:
            rows.append({
                "Asset": meta["label"],
                "Latest": np.nan,
                "1-Step %": np.nan,
                "Z-score(60)": np.nan,
                "Range Pos(52W)": np.nan,
                "From High": np.nan,
                "Interpretation": "No valid close data",
            })
            continue

        latest = valid.iloc[-1]
        prev = valid.iloc[-2] if len(valid) > 1 else np.nan
        delta_pct = (latest / prev - 1.0) * 100.0 if pd.notna(prev) and prev != 0 else np.nan

        z60 = calc_zscore(valid, window=60)
        range_pos = calc_range_position(valid, window=min(252, period_days))
        dd_high = calc_drawdown_from_high(valid, window=min(252, period_days))

        rows.append({
            "Asset": meta["label"],
            "Latest": round(float(latest), 2) if pd.notna(latest) else np.nan,
            "1-Step %": round(float(delta_pct), 2) if pd.notna(delta_pct) else np.nan,
            "Z-score(60)": round(float(z60), 2) if z60 is not None else np.nan,
            "Range Pos(52W)": round(float(range_pos), 1) if range_pos is not None else np.nan,
            "From High": round(float(dd_high), 1) if dd_high is not None else np.nan,
            "Interpretation": meta["interpretation"],
        })

    return pd.DataFrame(rows)


def render_market_chart(data_dict: Dict[str, pd.DataFrame], key: str):
    meta = MARKET_TICKERS[key]

    if key not in data_dict or data_dict[key].empty:
        st.warning(f"{meta['label']} data not available.")
        return

    df = normalize_time_column(data_dict[key])
    df = ensure_numeric_column(df, "Close")

    st.markdown(f"### {meta['label']}")
    fig = make_line_chart(
        df=df,
        x_col="Time",
        y_col="Close",
        title=meta["label"],
        y_title=meta["y_title"],
        show_ma=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_debug:
        with st.expander(f"Debug: {meta['label']}"):
            st.write("Columns:", df.columns.tolist())
            st.dataframe(df.head(), use_container_width=True)


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

    market_period = st.selectbox(
        "Market chart period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=4,
    )

    market_interval = st.selectbox(
        "Market chart interval",
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
market_data = download_market_data(period=market_period, interval=market_interval)
fred_data = load_all_fred_data() if FRED_API_KEY else {}

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# Render functions
# ============================================================
def render_market_tab():
    st.subheader("💱 FX & Macro Market Monitoring")

    sub_fx, sub_macro = st.tabs(["FX", "Macro Assets"])

    with sub_fx:
        fx_keys = ["USDKRW", "EURKRW", "EURUSD", "DXY"]

        st.markdown("### FX Summary")
        fx_summary = build_market_summary_table(market_data, fx_keys, market_period)
        st.dataframe(
            fx_summary[["Asset", "Latest", "1-Step %", "Z-score(60)", "Range Pos(52W)", "From High"]],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("FX Interpretation"):
            st.dataframe(
                fx_summary[["Asset", "Interpretation"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### FX Charts")
        col1, col2 = st.columns(2)
        with col1:
            render_market_chart(market_data, "USDKRW")
            render_market_chart(market_data, "EURUSD")
        with col2:
            render_market_chart(market_data, "EURKRW")
            render_market_chart(market_data, "DXY")

    with sub_macro:
        macro_keys = ["BTCEUR", "WTI", "BRENT", "GOLD", "SILVER", "NATGAS"]

        st.markdown("### Macro Asset Summary")
        macro_summary = build_market_summary_table(market_data, macro_keys, market_period)
        st.dataframe(
            macro_summary[["Asset", "Latest", "1-Step %", "Z-score(60)", "Range Pos(52W)", "From High"]],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Macro Asset Interpretation"):
            st.dataframe(
                macro_summary[["Asset", "Interpretation"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Macro Asset Charts")
        col1, col2 = st.columns(2)
        with col1:
            render_market_chart(market_data, "BTCEUR")
            render_market_chart(market_data, "WTI")
            render_market_chart(market_data, "GOLD")
        with col2:
            render_market_chart(market_data, "BRENT")
            render_market_chart(market_data, "SILVER")
            render_market_chart(market_data, "NATGAS")


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

Suggested integration point for your earlier FRED dashboard:
- Liquidity block
- Rates block
- Credit block
- Inflation block
- Recession warning block
- Asset allocation implication
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
    ["💱 FX & Macro Market", "🏦 FRED Macro Analysis", "📈 FRED Detail Charts"]
)

with tab1:
    render_market_tab()

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

1. Add mini comparison charts
   - BTC vs Gold
   - WTI vs Brent
   - DXY vs USD/KRW

2. Add correlation block
   - 30D rolling correlation
   - 90D rolling correlation

3. Add alert thresholds
   - USD/KRW alert
   - EUR/KRW alert
   - Oil spike alert
   - DXY breakout alert

4. Add relative performance summary
   - 1M / 3M / 6M / 1Y returns table

5. Add Germany / Korea interpretation cards
   - KRW weakness warning
   - EUR strength and living cost pressure
   - Oil price and inflation burden

6. Merge your previous FRED dashboard blocks
   - Liquidity
   - Rates
   - Credit
   - Inflation
   - Recession
"""
    )
