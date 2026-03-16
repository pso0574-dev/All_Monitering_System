# streamlit_app.py
# ============================================================
# Global Market Overview Dashboard
# Robust version:
# - fixes blank charts caused by yfinance column parsing issues
# - per-series normalization
# - adjustable summary font sizes
# - adjustable metric cards per row
# - safer chart rendering with no-data handling
#
# Install:
#   pip install streamlit yfinance pandas numpy plotly requests
#
# Run:
#   streamlit run streamlit_app.py
#
# Optional:
#   Set FRED_API_KEY in environment variable
#   or .streamlit/secrets.toml
#   FRED_API_KEY="YOUR_API_KEY"
# ============================================================

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Global Market Overview Dashboard",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Global Market Overview Dashboard")
st.caption("Integrated monitoring for equities, FX, rates, credit, commodities, and Korea-focused signals")


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Display Settings")

lookback_days = st.sidebar.selectbox(
    "Price history window",
    options=[180, 365, 730, 1095],
    index=2,
)

refresh = st.sidebar.button("🔄 Refresh Data")

plot_theme = st.sidebar.selectbox(
    "Plot theme",
    options=["plotly_dark", "plotly_white"],
    index=0,
)

base_font_size = st.sidebar.slider("Base font size", 10, 24, 14)
title_font_size = st.sidebar.slider("Chart title font size", 14, 32, 22)
axis_font_size = st.sidebar.slider("Axis font size", 10, 24, 14)
legend_font_size = st.sidebar.slider("Legend font size", 10, 22, 12)

metric_label_size = st.sidebar.slider("Summary label font size", 10, 26, 15)
metric_value_size = st.sidebar.slider("Summary value font size", 18, 52, 30)
metric_delta_size = st.sidebar.slider("Summary delta font size", 10, 24, 13)

chart_height = st.sidebar.slider("Default chart height", 320, 900, 460)
metric_cards_per_row = st.sidebar.slider("Summary cards per row", 3, 8, 5)

show_moving_avg = st.sidebar.checkbox("Show moving averages", value=True)
show_drawdown = st.sidebar.checkbox("Show drawdown chart", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Optional FRED")

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
if not FRED_API_KEY:
    try:
        FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        FRED_API_KEY = ""

if FRED_API_KEY:
    st.sidebar.success("FRED API key detected")
else:
    st.sidebar.warning("No FRED API key found. Rates / macro panel will be limited.")


# ============================================================
# CSS for metric font sizes
# ============================================================
st.markdown(
    f"""
    <style>
    div[data-testid="metric-container"] {{
        padding: 8px 10px;
        border-radius: 12px;
    }}
    div[data-testid="metric-container"] label {{
        font-size: {metric_label_size}px !important;
        white-space: normal !important;
        line-height: 1.15 !important;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-size: {metric_value_size}px !important;
        line-height: 1.0 !important;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
        font-size: {metric_delta_size}px !important;
        line-height: 1.0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Universe
# ============================================================
MARKET_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "Euro Stoxx 50": "^STOXX50E",
    "DAX": "^GDAXI",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "SOX": "^SOX",
    "US Dollar Index": "DX-Y.NYB",
    "USD/KRW": "KRW=X",
    "EUR/KRW": "EURKRW=X",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CNY": "CNY=X",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "WTI": "CL=F",
    "Brent": "BZ=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Bitcoin": "BTC-USD",
    "TLT": "TLT",
    "IEF": "IEF",
    "TIP": "TIP",
    "DBC": "DBC",
    "GLD": "GLD",
    "HYG": "HYG",
    "LQD": "LQD",
    "EEM": "EEM",
    "EWY": "EWY",
    "EWG": "EWG",
    "FXE": "FXE",
    "UUP": "UUP",
}

FRED_SERIES = {
    "US 2Y": "DGS2",
    "US 10Y": "DGS10",
    "US 30Y": "DGS30",
    "US 3M": "DGS3MO",
    "US Real 10Y": "DFII10",
    "Fed Funds Rate": "FEDFUNDS",
    "US HY OAS": "BAMLH0A0HYM2",
    "US IG OAS": "BAMLC0A0CM",
    "TED Spread": "TEDRATE",
}


# ============================================================
# Helpers
# ============================================================
def format_number(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


def format_pct(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}%"


def safe_pct(a: float, b: float) -> float:
    try:
        if b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return (a / b - 1.0) * 100.0
    except Exception:
        return np.nan


def get_last_valid(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def compute_drawdown(price: pd.Series) -> pd.Series:
    if price.empty:
        return price
    running_max = price.cummax()
    return (price / running_max - 1.0) * 100.0


def compute_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean_ = series.rolling(window).mean()
    std_ = series.rolling(window).std()
    return (series - mean_) / std_


def calc_change_windows(s: pd.Series) -> Dict[str, float]:
    s = s.dropna()
    if s.empty:
        return {"1d": np.nan, "1w": np.nan, "1m": np.nan, "3m": np.nan, "1y": np.nan}

    n = len(s)
    return {
        "1d": safe_pct(s.iloc[-1], s.iloc[-2]) if n >= 2 else np.nan,
        "1w": safe_pct(s.iloc[-1], s.iloc[-6]) if n >= 6 else np.nan,
        "1m": safe_pct(s.iloc[-1], s.iloc[-22]) if n >= 22 else np.nan,
        "3m": safe_pct(s.iloc[-1], s.iloc[-66]) if n >= 66 else np.nan,
        "1y": safe_pct(s.iloc[-1], s.iloc[-252]) if n >= 252 else np.nan,
    }


def calc_percentile_position(s: pd.Series, window: int = 252) -> float:
    s = s.dropna()
    if len(s) < 20:
        return np.nan
    tail = s.iloc[-window:] if len(s) >= window else s
    min_v = tail.min()
    max_v = tail.max()
    cur = tail.iloc[-1]
    if max_v == min_v:
        return 50.0
    return (cur - min_v) / (max_v - min_v) * 100.0


def normalize_each_column(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            out[col] = np.nan
            continue
        first_valid = s.iloc[0]
        if pd.isna(first_valid) or first_valid == 0:
            out[col] = np.nan
            continue
        out[col] = df[col] / first_valid * 100.0
    return out


def get_plot_layout(title: str, yaxis_title: str = "", height: Optional[int] = None) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=title_font_size)),
        template=plot_theme,
        height=height or chart_height,
        margin=dict(l=50, r=30, t=80, b=50),
        font=dict(size=base_font_size),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=legend_font_size),
        ),
        xaxis=dict(
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=axis_font_size),
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title=yaxis_title,
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=axis_font_size),
            showgrid=True,
            zeroline=False,
        ),
    )


def classify_regime(vix_level: float, usdkrw_1m: float, spx_1m: float, wti_1m: float) -> str:
    score = 0

    if not pd.isna(vix_level):
        if vix_level > 25:
            score += 2
        elif vix_level > 18:
            score += 1

    if not pd.isna(usdkrw_1m):
        if usdkrw_1m > 3:
            score += 2
        elif usdkrw_1m > 1:
            score += 1

    if not pd.isna(spx_1m):
        if spx_1m < -5:
            score += 2
        elif spx_1m < -2:
            score += 1

    if not pd.isna(wti_1m):
        if wti_1m > 8:
            score += 1

    if score >= 5:
        return "🔴 Risk-off / Stress"
    elif score >= 3:
        return "🟠 Cautious / Inflation Pressure"
    else:
        return "🟢 Balanced / Risk-on"


def show_metrics_in_rows(items: List[tuple], per_row: int = 5) -> None:
    for i in range(0, len(items), per_row):
        row_items = items[i:i + per_row]
        cols = st.columns(len(row_items))
        for c, (label, value, delta) in zip(cols, row_items):
            c.metric(label, value, delta)


def build_nonempty_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    out = []
    for c in candidates:
        if c in df.columns and not df[c].dropna().empty:
            out.append(c)
    return out


# ============================================================
# Data loaders
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_data(
    tickers: Dict[str, str],
    period_days: int = 730
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    end = datetime.today()
    start = end - timedelta(days=period_days)

    ticker_list = list(tickers.values())

    raw = yf.download(
        tickers=ticker_list,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    close = pd.DataFrame(index=raw.index)
    volume = pd.DataFrame(index=raw.index)

    inverse_map = {v: k for k, v in tickers.items()}

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0).unique())
        level1 = list(raw.columns.get_level_values(1).unique())

        # Pattern A: (Ticker, Field)
        if any(t in level0 for t in ticker_list):
            for ticker in ticker_list:
                label = inverse_map.get(ticker, ticker)

                if (ticker, "Close") in raw.columns:
                    close[label] = raw[(ticker, "Close")]
                elif (ticker, "Adj Close") in raw.columns:
                    close[label] = raw[(ticker, "Adj Close")]

                if (ticker, "Volume") in raw.columns:
                    volume[label] = raw[(ticker, "Volume")]

        # Pattern B: (Field, Ticker)
        elif "Close" in level0 or "Adj Close" in level0:
            field_close = "Close" if "Close" in level0 else "Adj Close"

            if field_close in raw.columns.get_level_values(0):
                tmp_close = raw[field_close].copy()
                tmp_close = tmp_close.rename(columns=inverse_map)
                for c in tmp_close.columns:
                    close[c] = tmp_close[c]

            if "Volume" in raw.columns.get_level_values(0):
                tmp_volume = raw["Volume"].copy()
                tmp_volume = tmp_volume.rename(columns=inverse_map)
                for c in tmp_volume.columns:
                    volume[c] = tmp_volume[c]

        # Pattern C fallback
        else:
            for ticker in ticker_list:
                label = inverse_map.get(ticker, ticker)
                try:
                    sub = raw[ticker]
                    if "Close" in sub.columns:
                        close[label] = sub["Close"]
                    elif "Adj Close" in sub.columns:
                        close[label] = sub["Adj Close"]

                    if "Volume" in sub.columns:
                        volume[label] = sub["Volume"]
                except Exception:
                    continue

    else:
        # Single ticker fallback
        only_label = list(tickers.keys())[0]
        if "Close" in raw.columns:
            close[only_label] = raw["Close"]
        elif "Adj Close" in raw.columns:
            close[only_label] = raw["Adj Close"]

        if "Volume" in raw.columns:
            volume[only_label] = raw["Volume"]

    close = close.sort_index().dropna(how="all")
    volume = volume.sort_index().dropna(how="all")

    ordered_close = [k for k in tickers.keys() if k in close.columns]
    ordered_volume = [k for k in tickers.keys() if k in volume.columns]

    close = close[ordered_close] if ordered_close else close
    volume = volume[ordered_volume] if ordered_volume else volume

    return close, volume


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str) -> pd.Series:
    if not FRED_API_KEY:
        return pd.Series(dtype=float)

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": (datetime.today() - timedelta(days=3650)).strftime("%Y-%m-%d"),
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if not obs:
            return pd.Series(dtype=float)

        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = pd.Series(df["value"].values, index=df["date"]).sort_index()
        return s
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fred_bundle(series_map: Dict[str, str]) -> Dict[str, pd.Series]:
    out = {}
    for name, sid in series_map.items():
        out[name] = fetch_fred_series(sid)
    return out


# ============================================================
# Refresh
# ============================================================
if refresh:
    st.cache_data.clear()


# ============================================================
# Load data
# ============================================================
with st.spinner("Loading market data..."):
    close_df, volume_df = fetch_yahoo_data(MARKET_TICKERS, period_days=lookback_days)
    fred_data = load_fred_bundle(FRED_SERIES) if FRED_API_KEY else {}

if close_df.empty:
    st.error("Failed to load Yahoo Finance data. Please retry.")
    st.stop()


# ============================================================
# Derived data
# ============================================================
norm_df = normalize_each_column(close_df).dropna(how="all")
drawdown_df = close_df.apply(compute_drawdown)
zscore_1y = close_df.apply(lambda s: compute_zscore(s, window=252))

snapshot_rows = []
for asset in close_df.columns:
    s = close_df[asset].dropna()
    if s.empty:
        continue

    chg = calc_change_windows(s)

    snapshot_rows.append({
        "Asset": asset,
        "Last": s.iloc[-1],
        "1D %": chg["1d"],
        "1W %": chg["1w"],
        "1M %": chg["1m"],
        "3M %": chg["3m"],
        "1Y %": chg["1y"],
        "52W Position %": calc_percentile_position(s),
        "Drawdown %": compute_drawdown(s).iloc[-1],
        "Z-score 1Y": zscore_1y[asset].dropna().iloc[-1] if not zscore_1y[asset].dropna().empty else np.nan,
    })

snapshot_df = pd.DataFrame(snapshot_rows)

usdkrw = close_df["USD/KRW"].dropna() if "USD/KRW" in close_df.columns else pd.Series(dtype=float)
eurkrw = close_df["EUR/KRW"].dropna() if "EUR/KRW" in close_df.columns else pd.Series(dtype=float)
spx = close_df["S&P 500"].dropna() if "S&P 500" in close_df.columns else pd.Series(dtype=float)
wti = close_df["WTI"].dropna() if "WTI" in close_df.columns else pd.Series(dtype=float)

usdkrw_1m = calc_change_windows(usdkrw)["1m"] if not usdkrw.empty else np.nan
spx_1m = calc_change_windows(spx)["1m"] if not spx.empty else np.nan
wti_1m = calc_change_windows(wti)["1m"] if not wti.empty else np.nan

vix_level = np.nan
try:
    vix_raw = yf.download("^VIX", period="2y", auto_adjust=True, progress=False, group_by="ticker")
    if not vix_raw.empty:
        if isinstance(vix_raw.columns, pd.MultiIndex):
            if ("^VIX", "Close") in vix_raw.columns:
                vix_level = float(vix_raw[("^VIX", "Close")].dropna().iloc[-1])
            elif ("Close", "^VIX") in vix_raw.columns:
                vix_level = float(vix_raw[("Close", "^VIX")].dropna().iloc[-1])
        elif "Close" in vix_raw.columns:
            vix_level = float(vix_raw["Close"].dropna().iloc[-1])
except Exception:
    pass

regime = classify_regime(vix_level, usdkrw_1m, spx_1m, wti_1m)


# ============================================================
# Chart functions
# ============================================================
def make_line_chart(df: pd.DataFrame, title: str, yaxis_title: str = "", height: Optional[int] = None) -> go.Figure:
    df = df.dropna(how="all")
    fig = go.Figure()

    added = 0
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=col,
            )
        )
        added += 1

    fig.update_layout(**get_plot_layout(title, yaxis_title, height))

    if added == 0:
        fig.add_annotation(
            text="No valid data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=title_font_size),
        )

    return fig


def make_single_price_chart(price: pd.Series, name: str, show_ma: bool = True, height: Optional[int] = None) -> go.Figure:
    s = price.dropna()
    fig = go.Figure()

    if not s.empty:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name))

        if show_ma:
            fig.add_trace(go.Scatter(x=s.index, y=s.rolling(20).mean(), mode="lines", name="MA20"))
            fig.add_trace(go.Scatter(x=s.index, y=s.rolling(50).mean(), mode="lines", name="MA50"))
            fig.add_trace(go.Scatter(x=s.index, y=s.rolling(200).mean(), mode="lines", name="MA200"))
    else:
        fig.add_annotation(
            text="No valid price data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=title_font_size),
        )

    fig.update_layout(**get_plot_layout(f"{name} Price Trend", "", height))
    return fig


def make_drawdown_chart(price: pd.Series, name: str, height: Optional[int] = None) -> go.Figure:
    dd = compute_drawdown(price).dropna()
    fig = go.Figure()

    if not dd.empty:
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                name=f"{name} Drawdown",
                fill="tozeroy",
            )
        )
    else:
        fig.add_annotation(
            text="No valid drawdown data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=title_font_size),
        )

    fig.update_layout(**get_plot_layout(f"{name} Drawdown vs Previous Peak", "Drawdown (%)", height or 340))
    return fig


def make_heatmap(df: pd.DataFrame, title: str, height: Optional[int] = None) -> go.Figure:
    plot_df = df.copy().dropna(how="all")
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(**get_plot_layout(title, "", height))
        fig.add_annotation(
            text="No valid heatmap data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=title_font_size),
        )
        return fig

    fig = px.imshow(
        plot_df,
        aspect="auto",
        text_auto=".1f",
        template=plot_theme,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=title_font_size)),
        height=height or chart_height,
        margin=dict(l=50, r=30, t=80, b=50),
        font=dict(size=base_font_size),
    )
    return fig


def make_yield_curve_chart(curve_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    valid = curve_df.dropna(subset=["Yield"])
    if not valid.empty:
        fig.add_trace(
            go.Scatter(
                x=valid["Maturity"],
                y=valid["Yield"],
                mode="lines+markers+text",
                text=[f"{v:.2f}%" for v in valid["Yield"]],
                textposition="top center",
                name="US Curve",
            )
        )
    else:
        fig.add_annotation(
            text="No valid yield curve data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=title_font_size),
        )

    fig.update_layout(**get_plot_layout("US Yield Curve", "Yield (%)", 420))
    fig.update_xaxes(title_text="Maturity (Years)")
    return fig


def make_dual_axis_chart(
    left_series: pd.Series,
    right_series: pd.Series,
    left_name: str,
    right_name: str,
    title: str,
    height: Optional[int] = None,
) -> go.Figure:
    aligned = pd.concat([left_series, right_series], axis=1).dropna()
    aligned.columns = [left_name, right_name]

    fig = go.Figure()

    if not aligned.empty:
        fig.add_trace(
            go.Scatter(
                x=aligned.index,
                y=aligned[left_name],
                mode="lines",
                name=left_name,
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=aligned.index,
                y=aligned[right_name],
                mode="lines",
                name=right_name,
                yaxis="y2",
            )
        )
    else:
        fig.add_annotation(
            text="No valid aligned data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=title_font_size),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=title_font_size)),
        template=plot_theme,
        height=height or chart_height,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=base_font_size),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=legend_font_size)),
        yaxis=dict(title=left_name, title_font=dict(size=axis_font_size), tickfont=dict(size=axis_font_size)),
        yaxis2=dict(
            title=right_name,
            overlaying="y",
            side="right",
            title_font=dict(size=axis_font_size),
            tickfont=dict(size=axis_font_size),
        ),
        xaxis=dict(tickfont=dict(size=axis_font_size)),
    )
    return fig


# ============================================================
# Top summary
# ============================================================
st.subheader("Market Regime")
st.info(regime)

top_assets = [
    "S&P 500",
    "Nasdaq 100",
    "Euro Stoxx 50",
    "KOSPI",
    "USD/KRW",
    "EUR/KRW",
    "US Dollar Index",
    "WTI",
    "Gold",
    "Bitcoin",
]

metric_items = []
for asset in top_assets:
    if asset in close_df.columns:
        s = close_df[asset].dropna()
        if not s.empty:
            chg1d = calc_change_windows(s)["1d"]
            metric_items.append((asset, format_number(s.iloc[-1], 2), format_pct(chg1d, 2)))

show_metrics_in_rows(metric_items, per_row=metric_cards_per_row)


# ============================================================
# Tabs
# ============================================================
tab_snapshot, tab_fx, tab_rates, tab_risk, tab_korea, tab_data = st.tabs([
    "📊 Snapshot",
    "💱 FX",
    "📉 Rates",
    "⚠️ Risk",
    "🇰🇷 Korea Focus",
    "🧾 Data Table",
])


# ============================================================
# Snapshot
# ============================================================
with tab_snapshot:
    st.subheader("Global Snapshot Overview")

    focus_assets = build_nonempty_columns(
        norm_df,
        [
            "S&P 500", "Nasdaq 100", "Euro Stoxx 50", "DAX", "Nikkei 225",
            "KOSPI", "US Dollar Index", "USD/KRW", "EUR/KRW", "WTI", "Gold", "Bitcoin"
        ],
    )

    if focus_assets:
        st.plotly_chart(
            make_line_chart(norm_df[focus_assets], "Normalized Relative Performance (Base = 100)", "Index Level"),
            use_container_width=True,
            key="snapshot_relative_perf",
        )
    else:
        st.warning("No valid normalized data available for snapshot chart.")

    st.markdown("### Snapshot Table")
    st.dataframe(
        snapshot_df.sort_values("Asset").style.format({
            "Last": "{:,.2f}",
            "1D %": "{:.2f}%",
            "1W %": "{:.2f}%",
            "1M %": "{:.2f}%",
            "3M %": "{:.2f}%",
            "1Y %": "{:.2f}%",
            "52W Position %": "{:.1f}",
            "Drawdown %": "{:.2f}%",
            "Z-score 1Y": "{:.2f}",
        }),
        use_container_width=True,
    )

    heat_assets = [
        "S&P 500", "Nasdaq 100", "Euro Stoxx 50", "KOSPI",
        "US Dollar Index", "USD/KRW", "EUR/KRW",
        "WTI", "Gold", "TLT", "TIP", "DBC", "Bitcoin"
    ]
    available_heat = [a for a in heat_assets if a in snapshot_df["Asset"].values]
    if available_heat:
        heat_df = snapshot_df.set_index("Asset").loc[available_heat, ["1D %", "1W %", "1M %", "3M %", "1Y %"]]
        st.plotly_chart(
            make_heatmap(heat_df, "Performance Heatmap"),
            use_container_width=True,
            key="snapshot_heatmap",
        )


# ============================================================
# FX
# ============================================================
with tab_fx:
    st.subheader("FX Monitoring")

    fx_assets = build_nonempty_columns(
        close_df,
        ["USD/KRW", "EUR/KRW", "EUR/USD", "USD/JPY", "USD/CNY", "US Dollar Index"]
    )

    col1, col2 = st.columns([1.35, 1.0])

    with col1:
        fx_norm_assets = build_nonempty_columns(norm_df, fx_assets)
        if fx_norm_assets:
            st.plotly_chart(
                make_line_chart(norm_df[fx_norm_assets], "FX Relative Performance (Base = 100)", "Normalized Level"),
                use_container_width=True,
                key="fx_relative_performance",
            )
        else:
            st.warning("FX normalized chart data is unavailable.")

    with col2:
        fx_summary_rows = []
        for a in fx_assets:
            s = close_df[a].dropna()
            if s.empty:
                continue
            chg = calc_change_windows(s)
            fx_summary_rows.append({
                "Pair": a,
                "Last": s.iloc[-1],
                "1D %": chg["1d"],
                "1M %": chg["1m"],
                "3M %": chg["3m"],
                "1Y %": chg["1y"],
                "52W Position %": calc_percentile_position(s),
                "Z-score 1Y": zscore_1y[a].dropna().iloc[-1] if not zscore_1y[a].dropna().empty else np.nan,
            })

        if fx_summary_rows:
            fx_summary_df = pd.DataFrame(fx_summary_rows)
            st.dataframe(
                fx_summary_df.style.format({
                    "Last": "{:,.2f}",
                    "1D %": "{:.2f}%",
                    "1M %": "{:.2f}%",
                    "3M %": "{:.2f}%",
                    "1Y %": "{:.2f}%",
                    "52W Position %": "{:.1f}",
                    "Z-score 1Y": "{:.2f}",
                }),
                use_container_width=True,
            )

    if fx_assets:
        fx_focus = st.selectbox("Select FX pair", options=fx_assets, index=0, key="fx_focus_select")
        price = close_df[fx_focus].dropna()

        left, right = st.columns([1.4, 1.0])

        with left:
            st.plotly_chart(
                make_single_price_chart(price, fx_focus, show_ma=show_moving_avg),
                use_container_width=True,
                key=f"fx_price_{fx_focus}",
            )

        with right:
            if show_drawdown:
                st.plotly_chart(
                    make_drawdown_chart(price, fx_focus),
                    use_container_width=True,
                    key=f"fx_dd_{fx_focus}",
                )

        st.markdown("### FX Interpretation Panel")

        m1, m2, m3, m4 = st.columns(4)
        latest = get_last_valid(price)
        pos52 = calc_percentile_position(price)
        z1y = zscore_1y[fx_focus].dropna().iloc[-1] if not zscore_1y[fx_focus].dropna().empty else np.nan
        ddv = compute_drawdown(price).dropna().iloc[-1] if not compute_drawdown(price).dropna().empty else np.nan

        m1.metric("Latest", format_number(latest, 2))
        m2.metric("52W Position", f"{format_number(pos52, 1)} / 100")
        m3.metric("1Y Z-score", format_number(z1y, 2))
        m4.metric("Drawdown", format_pct(ddv, 2))

        helper_rows = []
        for helper in ["US Dollar Index", "S&P 500", "KOSPI", "WTI", "Gold"]:
            if helper in close_df.columns:
                aligned = close_df[[fx_focus, helper]].dropna()
                if len(aligned) > 20:
                    corr = aligned.pct_change().dropna().corr().iloc[0, 1]
                    helper_rows.append({"Driver": helper, "Return Correlation": corr})

        if helper_rows:
            helper_df = pd.DataFrame(helper_rows)
            st.dataframe(
                helper_df.style.format({"Return Correlation": "{:.2f}"}),
                use_container_width=True,
            )


# ============================================================
# Rates
# ============================================================
with tab_rates:
    st.subheader("Rates & Yield Curve")

    if fred_data:
        us2y = get_last_valid(fred_data.get("US 2Y", pd.Series(dtype=float)))
        us10y = get_last_valid(fred_data.get("US 10Y", pd.Series(dtype=float)))
        us30y = get_last_valid(fred_data.get("US 30Y", pd.Series(dtype=float)))
        us3m = get_last_valid(fred_data.get("US 3M", pd.Series(dtype=float)))
        real10y = get_last_valid(fred_data.get("US Real 10Y", pd.Series(dtype=float)))

        c1, c2 = st.columns([1.2, 1.0])

        with c1:
            curve_df = pd.DataFrame({
                "Maturity": [0.25, 2, 10, 30],
                "Yield": [us3m, us2y, us10y, us30y],
            })
            st.plotly_chart(
                make_yield_curve_chart(curve_df),
                use_container_width=True,
                key="rates_yield_curve",
            )

        with c2:
            spread_10_2 = us10y - us2y if not pd.isna(us10y) and not pd.isna(us2y) else np.nan
            spread_10_3m = us10y - us3m if not pd.isna(us10y) and not pd.isna(us3m) else np.nan

            r1, r2 = st.columns(2)
            r1.metric("US 2Y", format_pct(us2y, 2))
            r2.metric("US 10Y", format_pct(us10y, 2))

            r3, r4 = st.columns(2)
            r3.metric("10Y - 2Y", format_pct(spread_10_2, 2))
            r4.metric("10Y - 3M", format_pct(spread_10_3m, 2))

            r5, r6 = st.columns(2)
            r5.metric("US 30Y", format_pct(us30y, 2))
            r6.metric("Real 10Y", format_pct(real10y, 2))

        rate_df = pd.DataFrame({
            k: fred_data[k]
            for k in ["US 2Y", "US 10Y", "US 30Y", "US Real 10Y", "Fed Funds Rate"]
            if k in fred_data and not fred_data[k].empty
        }).dropna(how="all")

        if not rate_df.empty:
            st.plotly_chart(
                make_line_chart(rate_df, "US Rates History", "Yield (%)"),
                use_container_width=True,
                key="rates_history",
            )

        if "TLT" in close_df.columns and "US 10Y" in rate_df.columns:
            st.plotly_chart(
                make_dual_axis_chart(close_df["TLT"], rate_df["US 10Y"], "TLT", "US 10Y Yield", "TLT vs US 10Y"),
                use_container_width=True,
                key="rates_tlt_dual",
            )
    else:
        st.info("FRED API key not detected. Showing ETF-based rate proxies only.")
        bond_assets = build_nonempty_columns(norm_df, ["TLT", "IEF", "TIP"])
        if bond_assets:
            st.plotly_chart(
                make_line_chart(norm_df[bond_assets], "Bond ETF Relative Performance", "Index Level"),
                use_container_width=True,
                key="rates_fallback_bond_etf",
            )


# ============================================================
# Risk
# ============================================================
with tab_risk:
    st.subheader("Risk / Credit / Commodities")

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("VIX", format_number(vix_level, 2))
    rc2.metric("USD/KRW 1M", format_pct(usdkrw_1m, 2))
    rc3.metric("S&P 500 1M", format_pct(spx_1m, 2))
    rc4.metric("WTI 1M", format_pct(wti_1m, 2))

    risk_assets = build_nonempty_columns(
        norm_df,
        ["S&P 500", "Nasdaq 100", "KOSPI", "WTI", "Gold", "TLT", "TIP", "DBC", "HYG", "LQD", "US Dollar Index"]
    )

    if risk_assets:
        st.plotly_chart(
            make_line_chart(norm_df[risk_assets], "Cross-Asset Risk Monitoring (Base = 100)", "Index Level"),
            use_container_width=True,
            key="risk_cross_asset",
        )
    else:
        st.warning("No valid risk monitoring data available.")

    left, right = st.columns([1.2, 1.0])

    with left:
        if fred_data:
            credit_df = pd.DataFrame({
                k: fred_data[k]
                for k in ["US HY OAS", "US IG OAS", "TED Spread"]
                if k in fred_data and not fred_data[k].empty
            }).dropna(how="all")

            if not credit_df.empty:
                st.plotly_chart(
                    make_line_chart(credit_df, "Credit Stress Indicators", "Spread / Level"),
                    use_container_width=True,
                    key="risk_credit_stress",
                )
        else:
            cred_assets = build_nonempty_columns(norm_df, ["HYG", "LQD"])
            if cred_assets:
                st.plotly_chart(
                    make_line_chart(norm_df[cred_assets], "Credit ETF Relative Performance", "Index Level"),
                    use_container_width=True,
                    key="risk_credit_etf",
                )

    with right:
        stress_rows = [
            {"Indicator": "VIX", "Value": vix_level},
            {"Indicator": "USD/KRW 1M Change", "Value": usdkrw_1m},
            {"Indicator": "S&P 500 1M Change", "Value": spx_1m},
            {"Indicator": "WTI 1M Change", "Value": wti_1m},
        ]

        if fred_data:
            stress_rows.append({"Indicator": "US HY OAS", "Value": get_last_valid(fred_data.get("US HY OAS", pd.Series(dtype=float)))})
            stress_rows.append({"Indicator": "US IG OAS", "Value": get_last_valid(fred_data.get("US IG OAS", pd.Series(dtype=float)))})

        stress_df = pd.DataFrame(stress_rows)
        st.dataframe(
            stress_df.style.format({"Value": "{:.2f}"}),
            use_container_width=True,
        )


# ============================================================
# Korea Focus
# ============================================================
with tab_korea:
    st.subheader("Korea Focus")

    korea_assets = build_nonempty_columns(
        norm_df,
        ["KOSPI", "KOSDAQ", "USD/KRW", "EUR/KRW", "EWY", "SOX", "US Dollar Index", "WTI", "Copper"]
    )

    if korea_assets:
        st.plotly_chart(
            make_line_chart(norm_df[korea_assets], "Korea-Focused Relative Performance (Base = 100)", "Index Level"),
            use_container_width=True,
            key="korea_relative_perf",
        )
    else:
        st.warning("No valid Korea-focused data available.")

    kc1, kc2 = st.columns([1.2, 1.0])

    with kc1:
        compare_options = build_nonempty_columns(
            norm_df,
            ["KOSPI", "USD/KRW", "SOX", "WTI", "Copper", "US Dollar Index"]
        )

        selected_compare = st.multiselect(
            "Select Korea-related comparison assets",
            options=compare_options,
            default=[a for a in ["KOSPI", "USD/KRW", "SOX"] if a in compare_options],
            key="korea_compare_multiselect",
        )

        if selected_compare:
            st.plotly_chart(
                make_line_chart(norm_df[selected_compare], "Selected Korea Monitoring Assets", "Index Level"),
                use_container_width=True,
                key="korea_selected_assets",
            )

    with kc2:
        korea_rows = []
        for a in ["KOSPI", "KOSDAQ", "USD/KRW", "EUR/KRW", "SOX", "WTI", "Copper"]:
            if a in close_df.columns and not close_df[a].dropna().empty:
                s = close_df[a].dropna()
                chg = calc_change_windows(s)
                korea_rows.append({
                    "Asset": a,
                    "Last": s.iloc[-1],
                    "1M %": chg["1m"],
                    "3M %": chg["3m"],
                    "1Y %": chg["1y"],
                    "Drawdown %": compute_drawdown(s).iloc[-1],
                })

        if korea_rows:
            korea_df = pd.DataFrame(korea_rows)
            st.dataframe(
                korea_df.style.format({
                    "Last": "{:,.2f}",
                    "1M %": "{:.2f}%",
                    "3M %": "{:.2f}%",
                    "1Y %": "{:.2f}%",
                    "Drawdown %": "{:.2f}%",
                }),
                use_container_width=True,
            )


# ============================================================
# Data Table
# ============================================================
with tab_data:
    st.subheader("Raw Monitoring Tables")

    st.dataframe(
        snapshot_df.sort_values("Asset").reset_index(drop=True).style.format({
            "Last": "{:,.2f}",
            "1D %": "{:.2f}%",
            "1W %": "{:.2f}%",
            "1M %": "{:.2f}%",
            "3M %": "{:.2f}%",
            "1Y %": "{:.2f}%",
            "52W Position %": "{:.1f}",
            "Drawdown %": "{:.2f}%",
            "Z-score 1Y": "{:.2f}",
        }),
        use_container_width=True,
    )

    st.markdown("### Debug: Loaded Market Columns")
    st.write(list(close_df.columns))

    st.markdown("### Debug: Non-empty Column Count")
    st.write(int(sum(~close_df.isna().all(axis=0))))

    csv = close_df.to_csv().encode("utf-8")
    st.download_button(
        "Download price history CSV",
        data=csv,
        file_name="global_market_overview_prices.csv",
        mime="text/csv",
        key="download_price_csv",
    )


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "This version fixes blank charts with robust yfinance parsing and adds adjustable summary font sizes. "
    "If some symbols are unavailable in your region, those series will be skipped automatically."
)
