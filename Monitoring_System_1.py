# streamlit_app.py
# ============================================================
# Personal Financial View Dashboard
# - Tab 1: FX & Macro Market
# - Tab 2: FRED Macro Analysis
# - Tab 3: FRED Detail Charts
#
# FX & Macro Market:
# - Summary table first
# - Charts below
# - USD/KRW, EUR/KRW, EUR/USD, DXY (fallback)
# - BTC/EUR, WTI, Brent, Gold, Silver, Natural Gas
#
# FRED Macro Analysis:
# - Overview
# - Liquidity
# - Rates
# - Credit
# - Inflation
# - Labor
# - Recession
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
from typing import Dict, Optional, Tuple, List

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
st.caption("FX monitoring + macro market assets + FRED macro risk dashboard")


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
        "interpretation": "Higher EUR/KRW can raise Germany-based living cost burden in KRW terms.",
    },
    "EURUSD": {
        "ticker": "EURUSD=X",
        "label": "EUR / USD",
        "y_title": "USD per EUR",
        "interpretation": "EUR/USD tracks broad EUR versus USD strength.",
    },
    "DXY": {
        "ticker": ["DX-Y.NYB", "DX=F", "UUP"],
        "label": "US Dollar Index / Proxy",
        "y_title": "Index / Proxy",
        "interpretation": "A stronger dollar proxy often tightens global financial conditions.",
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
        "interpretation": "Silver mixes precious-metal behavior with industrial demand.",
    },
    "NATGAS": {
        "ticker": "NG=F",
        "label": "Natural Gas",
        "y_title": "USD",
        "interpretation": "Natural gas is important for energy-cost and industrial demand monitoring.",
    },
}

FRED_SERIES = {
    "WALCL": {
        "name": "Fed Total Assets",
        "category": "Liquidity",
        "description": "Fed balance sheet size. Rising assets often imply more liquidity support.",
    },
    "RRPONTSYD": {
        "name": "Overnight Reverse Repo",
        "category": "Liquidity",
        "description": "High reverse repo usage can reflect excess money-market liquidity parking.",
    },
    "WTREGEN": {
        "name": "Treasury General Account",
        "category": "Liquidity",
        "description": "TGA changes can drain or inject market liquidity depending on direction.",
    },
    "M2SL": {
        "name": "M2 Money Supply",
        "category": "Liquidity",
        "description": "Money supply trend is a broad liquidity backdrop for risk assets.",
    },
    "FEDFUNDS": {
        "name": "Fed Funds Rate",
        "category": "Rates",
        "description": "Higher policy rates reduce liquidity and increase financing costs.",
    },
    "DGS2": {
        "name": "US 2Y Treasury Yield",
        "category": "Rates",
        "description": "Short-end rates reflect policy expectations and tightening pressure.",
    },
    "DGS10": {
        "name": "US 10Y Treasury Yield",
        "category": "Rates",
        "description": "Long-end yields matter for valuation multiples and discount rates.",
    },
    "T10Y2Y": {
        "name": "10Y - 2Y Treasury Spread",
        "category": "Recession",
        "description": "Yield curve inversion is a classic recession warning signal.",
    },
    "BAMLH0A0HYM2": {
        "name": "High Yield OAS",
        "category": "Credit",
        "description": "Wider high-yield spreads suggest financial stress and risk aversion.",
    },
    "BAMLC0A0CM": {
        "name": "US Corporate Master OAS",
        "category": "Credit",
        "description": "Corporate credit spread gauge for broad financing conditions.",
    },
    "CPIAUCSL": {
        "name": "CPI (All Urban Consumers)",
        "category": "Inflation",
        "description": "Inflation pressure affects rates, real returns, and policy stance.",
    },
    "PCEPI": {
        "name": "PCE Price Index",
        "category": "Inflation",
        "description": "PCE is a key Fed inflation gauge.",
    },
    "UNRATE": {
        "name": "US Unemployment Rate",
        "category": "Labor",
        "description": "Rising unemployment can indicate weakening macro conditions.",
    },
    "PAYEMS": {
        "name": "Nonfarm Payrolls",
        "category": "Labor",
        "description": "Payroll growth helps track labor momentum and economic resilience.",
    },
    "ICSA": {
        "name": "Initial Jobless Claims",
        "category": "Labor",
        "description": "Jobless claims are a timely labor stress indicator.",
    },
    "USREC": {
        "name": "US Recession Indicator",
        "category": "Recession",
        "description": "NBER recession indicator series.",
    },
}

CATEGORY_ORDER = ["Liquidity", "Rates", "Credit", "Inflation", "Labor", "Recession"]


# ============================================================
# Helpers
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
    if score <= 4:
        return "Low Risk", "🟢"
    if score <= 9:
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


def determine_series_freq(series_id: str) -> str:
    weekly = {"WALCL", "RRPONTSYD", "ICSA"}
    monthly = {"M2SL", "UNRATE", "PAYEMS", "CPIAUCSL", "PCEPI", "USREC"}
    daily = {"FEDFUNDS", "DGS2", "DGS10", "T10Y2Y", "BAMLH0A0HYM2", "BAMLC0A0CM", "WTREGEN"}
    if series_id in weekly:
        return "weekly"
    if series_id in monthly:
        return "monthly"
    if series_id in daily:
        return "daily"
    return "unknown"


def compute_yoy_change(series: pd.Series, series_id: str) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    freq = determine_series_freq(series_id)
    if freq == "monthly":
        periods = min(12, len(s) - 1)
    elif freq == "weekly":
        periods = min(52, len(s) - 1)
    else:
        periods = min(252, len(s) - 1)
    return safe_pct_change(s, periods=periods)


def compute_short_change(series: pd.Series, series_id: str) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    freq = determine_series_freq(series_id)
    if freq == "monthly":
        periods = min(1, len(s) - 1)
    elif freq == "weekly":
        periods = min(4, len(s) - 1)
    else:
        periods = min(21, len(s) - 1)
    return safe_pct_change(s, periods=periods)


# ============================================================
# Data loading
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def download_market_data(period: str = "2y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    all_download_tickers = []
    for meta in MARKET_TICKERS.values():
        tk = meta["ticker"]
        if isinstance(tk, list):
            all_download_tickers.extend(tk)
        else:
            all_download_tickers.append(tk)

    all_download_tickers = list(dict.fromkeys(all_download_tickers))

    try:
        raw = yf.download(
            tickers=all_download_tickers,
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
    if not isinstance(raw.columns, pd.MultiIndex):
        return result

    level0 = raw.columns.get_level_values(0)

    for key, meta in MARKET_TICKERS.items():
        ticker_candidates = meta["ticker"] if isinstance(meta["ticker"], list) else [meta["ticker"]]
        selected_df = None

        for tk in ticker_candidates:
            if tk in level0:
                df = raw[tk].copy()
                df.columns = [str(c) for c in df.columns]
                if "Close" in df.columns:
                    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                    if df["Close"].dropna().shape[0] > 0:
                        selected_df = df
                        break

        if selected_df is not None:
            result[key] = selected_df

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


def make_category_bar_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=320, title="No category data")
        return fig

    grouped = (
        df.groupby("Category", as_index=False)["Risk Score"]
        .mean()
        .sort_values("Risk Score", ascending=False)
    )

    fig = go.Figure(
        go.Bar(
            x=grouped["Category"],
            y=grouped["Risk Score"],
            text=np.round(grouped["Risk Score"], 2),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Category Risk Score",
        xaxis_title="Category",
        yaxis_title="Average Risk Score",
        template="plotly_white",
        height=360,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


# ============================================================
# FRED logic
# ============================================================
def compute_indicator_risk(series_id: str, df: pd.DataFrame) -> Dict[str, object]:
    meta = FRED_SERIES[series_id]
    s = pd.to_numeric(df["value"], errors="coerce").dropna()

    if len(s) < 12:
        return {
            "Series ID": series_id,
            "Indicator": meta["name"],
            "Category": meta["category"],
            "Latest": np.nan,
            "Short %Chg": np.nan,
            "YoY %Chg": np.nan,
            "Trend": "N/A",
            "Risk Score": 0,
            "Comment": "Not enough data",
        }

    latest = float(s.iloc[-1])
    short_change = compute_short_change(s, series_id)
    yoy_change = compute_yoy_change(s, series_id)

    recent_window = s.iloc[-min(len(s), 24):]
    q25 = recent_window.quantile(0.25)
    q75 = recent_window.quantile(0.75)
    med = recent_window.median()

    risk_score = 0
    comment = meta["description"]

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
        comment = "Wider HY spread indicates rising credit stress."

    elif series_id == "BAMLC0A0CM":
        if latest > 3.5:
            risk_score = 2
        elif latest > 2.5:
            risk_score = 1
        else:
            risk_score = 0
        comment = "Broader corporate credit stress gauge."

    elif series_id == "UNRATE":
        if latest > q75:
            risk_score = 2
        elif latest > med:
            risk_score = 1
        else:
            risk_score = 0
        comment = "Rising unemployment suggests macro weakening."

    elif series_id == "ICSA":
        if latest > q75:
            risk_score = 2
        elif latest > med:
            risk_score = 1
        else:
            risk_score = 0
        comment = "Higher jobless claims can warn of labor stress."

    elif series_id == "CPIAUCSL":
        yoy = 0 if yoy_change is None else yoy_change
        if yoy > 4.0:
            risk_score = 2
        elif yoy > 2.5:
            risk_score = 1
        else:
            risk_score = 0
        comment = f"CPI inflation YoY: {format_num(yoy)}%."

    elif series_id == "PCEPI":
        yoy = 0 if yoy_change is None else yoy_change
        if yoy > 3.5:
            risk_score = 2
        elif yoy > 2.3:
            risk_score = 1
        else:
            risk_score = 0
        comment = f"PCE inflation YoY: {format_num(yoy)}%."

    elif series_id == "FEDFUNDS":
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

    elif series_id == "DGS2":
        if latest > q75:
            risk_score = 1
        else:
            risk_score = 0
        comment = "Higher short yields reflect tighter policy expectations."

    elif series_id == "M2SL":
        yoy = 0 if yoy_change is None else yoy_change
        if yoy < 0:
            risk_score = 2
        elif yoy < 3.0:
            risk_score = 1
        else:
            risk_score = 0
        comment = f"M2 YoY liquidity backdrop: {format_num(yoy)}%."

    elif series_id == "WALCL":
        short = 0 if short_change is None else short_change
        if short < -2.0:
            risk_score = 2
        elif short < 0:
            risk_score = 1
        else:
            risk_score = 0
        comment = f"Fed balance sheet short-term change: {format_num(short)}%."

    elif series_id == "RRPONTSYD":
        short = 0 if short_change is None else short_change
        if latest > q75 and short > 0:
            risk_score = 1
        else:
            risk_score = 0
        comment = "Elevated reverse repo can reflect excess liquidity parking."

    elif series_id == "WTREGEN":
        short = 0 if short_change is None else short_change
        if short > 10:
            risk_score = 1
            comment = "Rising TGA can drain liquidity from markets."
        else:
            risk_score = 0
            comment = "TGA not showing material liquidity drain."

    elif series_id == "PAYEMS":
        yoy = 0 if yoy_change is None else yoy_change
        if yoy < 0:
            risk_score = 2
        elif yoy < 1.0:
            risk_score = 1
        else:
            risk_score = 0
        comment = f"Payroll growth YoY: {format_num(yoy)}%."

    elif series_id == "USREC":
        risk_score = 2 if latest >= 1 else 0
        comment = "Official recession indicator."

    return {
        "Series ID": series_id,
        "Indicator": meta["name"],
        "Category": meta["category"],
        "Latest": latest,
        "Short %Chg": short_change,
        "YoY %Chg": yoy_change,
        "Trend": momentum_label(short_change),
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
    if total_score <= 4:
        timing_view = "Risk-on / constructive backdrop"
    elif total_score <= 9:
        timing_view = "Balanced / selective allocation"
    else:
        timing_view = "Defensive / risk management priority"

    return score_df, total_score, label, f"{icon} {timing_view}"


def filter_category(score_df: pd.DataFrame, category: str) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()
    return score_df[score_df["Category"] == category].copy()


# ============================================================
# Market summary helpers
# ============================================================
def build_market_summary_table(data_dict: Dict[str, pd.DataFrame], keys: List[str], period_label: str) -> pd.DataFrame:
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


def render_market_chart(data_dict: Dict[str, pd.DataFrame], key: str, show_debug: bool):
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
            st.write("Ticker source:", meta["ticker"])
            st.write("Columns:", df.columns.tolist())
            st.write("Valid close count:", int(df["Close"].dropna().shape[0]) if "Close" in df.columns else 0)
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

    fred_start_date = st.selectbox(
        "FRED start date",
        options=["2000-01-01", "2008-01-01", "2010-01-01", "2015-01-01", "2020-01-01"],
        index=2,
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
# Reload helper for start date
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_fred_data_with_start(start_date: str) -> Dict[str, pd.DataFrame]:
    return {sid: fetch_fred_series(sid, start_date=start_date) for sid in FRED_SERIES.keys()}


# ============================================================
# Load data
# ============================================================
market_data = download_market_data(period=market_period, interval=market_interval)
fred_data = load_all_fred_data_with_start(fred_start_date) if FRED_API_KEY else {}

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
            render_market_chart(market_data, "USDKRW", show_debug)
            render_market_chart(market_data, "EURUSD", show_debug)
        with col2:
            render_market_chart(market_data, "EURKRW", show_debug)
            render_market_chart(market_data, "DXY", show_debug)

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
            render_market_chart(market_data, "BTCEUR", show_debug)
            render_market_chart(market_data, "WTI", show_debug)
            render_market_chart(market_data, "GOLD", show_debug)
        with col2:
            render_market_chart(market_data, "BRENT", show_debug)
            render_market_chart(market_data, "SILVER", show_debug)
            render_market_chart(market_data, "NATGAS", show_debug)


def render_fred_category_section(score_df: pd.DataFrame, fred_data: Dict[str, pd.DataFrame], category: str):
    cat_df = filter_category(score_df, category)
    st.markdown(f"### {category} Snapshot")

    if cat_df.empty:
        st.info(f"No {category} data available.")
        return

    st.dataframe(
        cat_df[["Indicator", "Latest", "Short %Chg", "YoY %Chg", "Trend", "Risk Score", "Comment"]],
        use_container_width=True,
        hide_index=True,
    )

    series_ids = cat_df["Series ID"].tolist()
    cols = st.columns(2)
    for idx, sid in enumerate(series_ids):
        with cols[idx % 2]:
            df = fred_data.get(sid, pd.DataFrame())
            if not df.empty:
                st.plotly_chart(make_fred_chart(df, FRED_SERIES[sid]["name"]), use_container_width=True)


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

    sub_tabs = st.tabs(["Overview", "Liquidity", "Rates", "Credit", "Inflation", "Labor", "Recession"])

    with sub_tabs[0]:
        st.markdown("### Macro Scorecard")
        st.dataframe(
            score_df[["Category", "Indicator", "Latest", "Short %Chg", "YoY %Chg", "Trend", "Risk Score", "Comment"]],
            use_container_width=True,
            hide_index=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(make_risk_bar_chart(score_df), use_container_width=True)
        with col_b:
            st.plotly_chart(make_category_bar_chart(score_df), use_container_width=True)

        category_scores = (
            score_df.groupby("Category", as_index=False)["Risk Score"]
            .mean()
            .sort_values("Category")
        )
        st.markdown("### Category Summary")
        st.dataframe(category_scores, use_container_width=True, hide_index=True)

        with st.expander("Macro Interpretation"):
            st.markdown(
                f"""
**Current regime:** {risk_label}  
**Investment timing view:** {timing_view}

Interpretation guide:
- **Liquidity:** shrinking liquidity is usually a headwind for equities and crypto.
- **Rates:** higher rates pressure valuation multiples and financing conditions.
- **Credit:** wider spreads imply rising stress.
- **Inflation:** sticky inflation reduces room for easing.
- **Labor:** weaker labor data can confirm slowdown.
- **Recession:** inversion and labor deterioration increase downturn risk.
"""
            )

    with sub_tabs[1]:
        render_fred_category_section(score_df, fred_data, "Liquidity")

    with sub_tabs[2]:
        render_fred_category_section(score_df, fred_data, "Rates")

    with sub_tabs[3]:
        render_fred_category_section(score_df, fred_data, "Credit")

    with sub_tabs[4]:
        render_fred_category_section(score_df, fred_data, "Inflation")

    with sub_tabs[5]:
        render_fred_category_section(score_df, fred_data, "Labor")

    with sub_tabs[6]:
        render_fred_category_section(score_df, fred_data, "Recession")


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
        short_pct = compute_short_change(df["value"], sid)
        yoy_pct = compute_yoy_change(df["value"], sid)

        st.markdown(f"### {meta['name']}")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
        c1.metric("Latest", format_num(latest, 2))
        c2.metric("Change", format_num(delta, 2) if delta is not None else "N/A")
        c3.metric("YoY %", format_pct(yoy_pct, 2))
        c4.markdown(f"**Why it matters:** {meta['description']}")

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

1. Add relative performance summary
   - 1M / 3M / 6M / 1Y returns table

2. Add cross-market comparison charts
   - DXY vs USD/KRW
   - Gold vs BTC
   - WTI vs Brent

3. Add rolling correlation panel
   - 30D / 90D correlation

4. Add alert thresholds
   - USD/KRW breakout
   - EUR/KRW breakout
   - HY spread spike
   - Yield curve inversion

5. Add portfolio implication section
   - Equity
   - Bonds
   - Gold
   - Commodities
   - Cash

6. Add Germany / Korea interpretation cards
   - KRW weakness warning
   - EUR strength and living cost pressure
   - Oil price and inflation burden
"""
    )
