# streamlit_app.py
# ============================================================
# Global Market Overview Monitoring Dashboard
# - Global snapshot
# - FX (USD/KRW, EUR/KRW)
# - Rates / Yield Curve
# - Risk / Credit / Commodities
# - Korea focus
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
import math
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Global Market Overview Dashboard",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Global Market Overview Dashboard")
st.caption(
    "Integrated monitoring view for equities, FX, rates, risk, commodities, and Korea-specific macro signals"
)

# ============================================================
# Config
# ============================================================
LOOKBACK_DAYS_DEFAULT = 730

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
if not FRED_API_KEY:
    try:
        FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        FRED_API_KEY = ""

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Settings")

lookback_days = st.sidebar.selectbox(
    "Price history window",
    options=[180, 365, 730, 1095],
    index=2,
)

refresh = st.sidebar.button("🔄 Refresh Data")

show_volume = st.sidebar.checkbox("Show volume where available", value=False)
show_moving_avg = st.sidebar.checkbox("Show moving averages", value=True)
show_drawdown = st.sidebar.checkbox("Show drawdown chart", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Optional FRED")
st.sidebar.caption("Macro/credit data becomes richer if FRED_API_KEY is available.")
if FRED_API_KEY:
    st.sidebar.success("FRED API key detected")
else:
    st.sidebar.warning("No FRED API key found. Market sections still work.")

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
    "USD/KRW": "KRW=X",         # Yahoo convention: KRW=X = USDKRW
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
    "VIX Proxy ETF": "VIXY",
}

# FRED series
FRED_SERIES = {
    "US 2Y": "DGS2",
    "US 10Y": "DGS10",
    "US 30Y": "DGS30",
    "US 3M": "DGS3MO",
    "US Real 10Y": "DFII10",
    "Fed Funds Rate": "FEDFUNDS",
    "ECB Deposit Rate Proxy": "ECBDFR",
    "Germany 10Y": "IRLTLT01DEM156N",
    "Korea CPI YoY": "KORCPIALLMINMEI",
    "US CPI YoY": "CPIAUCSL",
    "US Core CPI": "CPILFESL",
    "US HY OAS": "BAMLH0A0HYM2",
    "US IG OAS": "BAMLC0A0CM",
    "MOVE Proxy": "MOVEINDEX",   # may not always be available
    "TED Spread": "TEDRATE",
    "US Recession Prob Proxy": "RECPROUSM156N",
    "Korea Exports YoY Proxy": "XTEXVA01KRM667S",
}

# ============================================================
# Helpers
# ============================================================
def safe_pct(a: float, b: float) -> float:
    try:
        if b in [0, None] or pd.isna(a) or pd.isna(b):
            return np.nan
        return (a / b - 1.0) * 100.0
    except Exception:
        return np.nan


def compute_drawdown(price: pd.Series) -> pd.Series:
    if price.empty:
        return price
    running_max = price.cummax()
    return (price / running_max - 1.0) * 100.0


def compute_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean_ = series.rolling(window).mean()
    std_ = series.rolling(window).std()
    return (series - mean_) / std_


def classify_regime(
    vix_level: Optional[float],
    usdkrw_chg_1m: Optional[float],
    spx_chg_1m: Optional[float],
    us10y_change_1m_bp: Optional[float],
    wti_chg_1m: Optional[float],
) -> str:
    score = 0

    if vix_level is not None and not pd.isna(vix_level):
        if vix_level > 25:
            score += 2
        elif vix_level > 18:
            score += 1

    if usdkrw_chg_1m is not None and not pd.isna(usdkrw_chg_1m):
        if usdkrw_chg_1m > 3:
            score += 2
        elif usdkrw_chg_1m > 1:
            score += 1

    if spx_chg_1m is not None and not pd.isna(spx_chg_1m):
        if spx_chg_1m < -5:
            score += 2
        elif spx_chg_1m < -2:
            score += 1

    if us10y_change_1m_bp is not None and not pd.isna(us10y_change_1m_bp):
        if us10y_change_1m_bp > 35:
            score += 1

    if wti_chg_1m is not None and not pd.isna(wti_chg_1m):
        if wti_chg_1m > 8:
            score += 1

    if score >= 5:
        return "🔴 Risk-off / Stress"
    elif score >= 3:
        return "🟠 Cautious / Inflation Pressure"
    else:
        return "🟢 Balanced / Risk-on"


def get_status_text(value: float, warn: float, danger: float, reverse: bool = False) -> str:
    if pd.isna(value):
        return "N/A"
    if not reverse:
        if value >= danger:
            return "🔴 High"
        if value >= warn:
            return "🟠 Moderate"
        return "🟢 Normal"
    else:
        if value <= danger:
            return "🔴 Weak"
        if value <= warn:
            return "🟠 Soft"
        return "🟢 Strong"


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


def get_series_last_valid(s: pd.Series) -> float:
    if s is None or s.empty:
        return np.nan
    s = s.dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def calc_change_windows(s: pd.Series) -> Dict[str, float]:
    s = s.dropna()
    if s.empty:
        return {"1d": np.nan, "1w": np.nan, "1m": np.nan, "3m": np.nan, "1y": np.nan}
    out = {}
    n = len(s)
    out["1d"] = safe_pct(s.iloc[-1], s.iloc[-2]) if n >= 2 else np.nan
    out["1w"] = safe_pct(s.iloc[-1], s.iloc[-6]) if n >= 6 else np.nan
    out["1m"] = safe_pct(s.iloc[-1], s.iloc[-22]) if n >= 22 else np.nan
    out["3m"] = safe_pct(s.iloc[-1], s.iloc[-66]) if n >= 66 else np.nan
    out["1y"] = safe_pct(s.iloc[-1], s.iloc[-252]) if n >= 252 else np.nan
    return out


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
    df = yf.download(
        ticker_list,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Normalize close
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"].copy()
        else:
            close = pd.DataFrame()
        if "Volume" in df.columns.get_level_values(0):
            volume = df["Volume"].copy()
        else:
            volume = pd.DataFrame()
    else:
        close = pd.DataFrame(index=df.index)
        volume = pd.DataFrame(index=df.index)
        if "Close" in df.columns:
            only_name = list(tickers.keys())[0]
            close[only_name] = df["Close"]
        if "Volume" in df.columns:
            only_name = list(tickers.keys())[0]
            volume[only_name] = df["Volume"]

    # Rename symbol -> label
    inverse_map = {v: k for k, v in tickers.items()}
    close = close.rename(columns=inverse_map)
    volume = volume.rename(columns=inverse_map)

    # Keep requested order if present
    close = close[[c for c in tickers.keys() if c in close.columns]]
    volume = volume[[c for c in tickers.keys() if c in volume.columns]]

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
        js = r.json()
        obs = js.get("observations", [])
        if not obs:
            return pd.Series(dtype=float)

        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = pd.Series(df["value"].values, index=df["date"])
        s = s.sort_index()
        return s
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fred_bundle(series_map: Dict[str, str]) -> Dict[str, pd.Series]:
    out = {}
    for name, sid in series_map.items():
        out[name] = fetch_fred_series(sid)
    return out


def build_yield_curve_df(fred_data: Dict[str, pd.Series]) -> pd.DataFrame:
    curve_labels = ["US 3M", "US 2Y", "US 10Y", "US 30Y"]
    maturity_years = [0.25, 2, 10, 30]
    values = []
    for lbl in curve_labels:
        values.append(get_series_last_valid(fred_data.get(lbl, pd.Series(dtype=float))))
    return pd.DataFrame({
        "Maturity": maturity_years,
        "Label": curve_labels,
        "Yield": values,
    })


# ============================================================
# Refresh hook
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
norm_df = close_df / close_df.iloc[0] * 100.0
drawdown_df = close_df.apply(compute_drawdown)

ma20 = close_df.rolling(20).mean()
ma50 = close_df.rolling(50).mean()
ma200 = close_df.rolling(200).mean()

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
        "52W Position %": calc_percentile_position(s, window=252),
        "Drawdown %": compute_drawdown(s).iloc[-1],
        "Z-score 1Y": zscore_1y[asset].dropna().iloc[-1] if not zscore_1y[asset].dropna().empty else np.nan,
    })

snapshot_df = pd.DataFrame(snapshot_rows)

# FX quick metrics
usdkrw = close_df["USD/KRW"].dropna() if "USD/KRW" in close_df.columns else pd.Series(dtype=float)
eurkrw = close_df["EUR/KRW"].dropna() if "EUR/KRW" in close_df.columns else pd.Series(dtype=float)
eurusd = close_df["EUR/USD"].dropna() if "EUR/USD" in close_df.columns else pd.Series(dtype=float)
dxy = close_df["US Dollar Index"].dropna() if "US Dollar Index" in close_df.columns else pd.Series(dtype=float)
spx = close_df["S&P 500"].dropna() if "S&P 500" in close_df.columns else pd.Series(dtype=float)
wti = close_df["WTI"].dropna() if "WTI" in close_df.columns else pd.Series(dtype=float)

usdkrw_1m = calc_change_windows(usdkrw).get("1m", np.nan) if not usdkrw.empty else np.nan
spx_1m = calc_change_windows(spx).get("1m", np.nan) if not spx.empty else np.nan
wti_1m = calc_change_windows(wti).get("1m", np.nan) if not wti.empty else np.nan

us10y = fred_data.get("US 10Y", pd.Series(dtype=float)).dropna() if fred_data else pd.Series(dtype=float)
us10y_change_1m_bp = np.nan
if not us10y.empty and len(us10y) >= 22:
    us10y_change_1m_bp = (us10y.iloc[-1] - us10y.iloc[-22]) * 100

vix_level = get_series_last_valid(close_df["VIX Proxy ETF"]) if "VIX Proxy ETF" in close_df.columns else np.nan
# Better proxy from actual VIX if available through Yahoo:
if "^VIX" not in MARKET_TICKERS.values():
    try:
        vix_raw = yf.download("^VIX", period="2y", auto_adjust=True, progress=False)
        if not vix_raw.empty and "Close" in vix_raw.columns:
            vix_level = float(vix_raw["Close"].dropna().iloc[-1])
    except Exception:
        pass

regime = classify_regime(vix_level, usdkrw_1m, spx_1m, us10y_change_1m_bp, wti_1m)

# ============================================================
# Chart functions
# ============================================================
def make_line_chart(
    df: pd.DataFrame,
    title: str,
    yaxis_title: str = "",
    height: int = 430,
) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        if df[col].dropna().empty:
            continue
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode="lines",
            name=col,
        ))
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=70, b=40),
        yaxis_title=yaxis_title,
    )
    return fig


def make_single_price_chart(
    price: pd.Series,
    name: str,
    volume: Optional[pd.Series] = None,
    show_ma: bool = True,
    height: int = 430,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=price.index,
        y=price.values,
        mode="lines",
        name=name,
    ))

    if show_ma:
        ma20_s = price.rolling(20).mean()
        ma50_s = price.rolling(50).mean()
        ma200_s = price.rolling(200).mean()
        fig.add_trace(go.Scatter(x=price.index, y=ma20_s, mode="lines", name="MA20"))
        fig.add_trace(go.Scatter(x=price.index, y=ma50_s, mode="lines", name="MA50"))
        fig.add_trace(go.Scatter(x=price.index, y=ma200_s, mode="lines", name="MA200"))

    fig.update_layout(
        title=f"{name} Price Trend",
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def make_drawdown_chart(price: pd.Series, name: str, height: int = 320) -> go.Figure:
    dd = compute_drawdown(price)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        mode="lines",
        name=f"{name} Drawdown",
        fill="tozeroy",
    ))
    fig.update_layout(
        title=f"{name} Drawdown vs Previous Peak",
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
        yaxis_title="Drawdown (%)",
    )
    return fig


def make_heatmap(df: pd.DataFrame, title: str, height: int = 420) -> go.Figure:
    plot_df = df.copy()
    fig = px.imshow(
        plot_df,
        aspect="auto",
        title=title,
        text_auto=".1f",
    )
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def make_yield_curve_chart(curve_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["Maturity"],
        y=curve_df["Yield"],
        mode="lines+markers+text",
        text=[f"{v:.2f}%" if not pd.isna(v) else "N/A" for v in curve_df["Yield"]],
        textposition="top center",
        name="US Curve",
    ))
    fig.update_layout(
        title="US Yield Curve",
        xaxis_title="Maturity (Years)",
        yaxis_title="Yield (%)",
        height=420,
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def make_dual_axis_chart(
    left_series: pd.Series,
    right_series: pd.Series,
    left_name: str,
    right_name: str,
    title: str,
    height: int = 430,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=left_series.index,
        y=left_series.values,
        mode="lines",
        name=left_name,
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=right_series.index,
        y=right_series.values,
        mode="lines",
        name=right_name,
        yaxis="y2",
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=70, b=40),
        yaxis=dict(title=left_name),
        yaxis2=dict(title=right_name, overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ============================================================
# Header summary
# ============================================================
st.subheader("Market Regime")
st.info(regime)

top_assets = [
    "S&P 500", "Nasdaq 100", "Euro Stoxx 50", "KOSPI",
    "USD/KRW", "EUR/KRW", "US Dollar Index", "WTI", "Gold", "Bitcoin"
]

metric_cols = st.columns(len(top_assets))
for i, asset in enumerate(top_assets):
    if asset in close_df.columns:
        s = close_df[asset].dropna()
        if not s.empty:
            chg1d = calc_change_windows(s)["1d"]
            metric_cols[i].metric(
                asset,
                format_number(s.iloc[-1], 2),
                format_pct(chg1d, 2)
            )

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
# Snapshot tab
# ============================================================
with tab_snapshot:
    st.subheader("Global Snapshot Overview")

    focus_assets = [
        "S&P 500", "Nasdaq 100", "Euro Stoxx 50", "DAX", "Nikkei 225", "KOSPI",
        "US Dollar Index", "USD/KRW", "EUR/KRW", "WTI", "Gold", "Bitcoin"
    ]
    available_focus = [a for a in focus_assets if a in norm_df.columns]
    if available_focus:
        st.plotly_chart(
            make_line_chart(norm_df[available_focus], "Normalized Relative Performance (Base = 100)", "Index Level"),
            use_container_width=True,
            key="snapshot_relative_perf"
        )

    st.markdown("### Snapshot Table")
    snapshot_show = snapshot_df.copy()
    numeric_cols = ["Last", "1D %", "1W %", "1M %", "3M %", "1Y %", "52W Position %", "Drawdown %", "Z-score 1Y"]
    for c in numeric_cols:
        if c in snapshot_show.columns:
            snapshot_show[c] = pd.to_numeric(snapshot_show[c], errors="coerce")

    st.dataframe(
        snapshot_show.sort_values("Asset").style.format({
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
        use_container_width=True
    )

    st.markdown("### Cross-Asset Heatmap")
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
            key="snapshot_heatmap"
        )

# ============================================================
# FX tab
# ============================================================
with tab_fx:
    st.subheader("FX Monitoring")

    fx_assets = ["USD/KRW", "EUR/KRW", "EUR/USD", "USD/JPY", "USD/CNY", "US Dollar Index"]
    fx_assets = [a for a in fx_assets if a in close_df.columns]

    col1, col2 = st.columns([1.3, 1.0])

    with col1:
        if fx_assets:
            st.plotly_chart(
                make_line_chart(norm_df[fx_assets], "FX Relative Performance (Base = 100)", "Normalized Level"),
                use_container_width=True,
                key="fx_relative_performance"
            )

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

        fx_summary_df = pd.DataFrame(fx_summary_rows)
        if not fx_summary_df.empty:
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
                use_container_width=True
            )

    fx_focus = st.selectbox(
        "Select FX pair",
        options=fx_assets,
        index=0 if fx_assets else None,
        key="fx_focus_select"
    )

    if fx_focus:
        price = close_df[fx_focus].dropna()
        volume = volume_df[fx_focus].dropna() if (show_volume and fx_focus in volume_df.columns) else None

        c1, c2 = st.columns([1.5, 1.0])
        with c1:
            st.plotly_chart(
                make_single_price_chart(price, fx_focus, volume=volume, show_ma=show_moving_avg),
                use_container_width=True,
                key=f"fx_price_{fx_focus}"
            )
        with c2:
            if show_drawdown:
                st.plotly_chart(
                    make_drawdown_chart(price, fx_focus),
                    use_container_width=True,
                    key=f"fx_dd_{fx_focus}"
                )

        # Interpretation helpers
        st.markdown("### FX Interpretation Panel")

        sub1, sub2, sub3, sub4 = st.columns(4)

        latest = get_series_last_valid(price)
        pos52 = calc_percentile_position(price)
        z1y = zscore_1y[fx_focus].dropna().iloc[-1] if not zscore_1y[fx_focus].dropna().empty else np.nan
        ddv = compute_drawdown(price).dropna().iloc[-1] if not compute_drawdown(price).dropna().empty else np.nan

        sub1.metric("Latest", format_number(latest, 2))
        sub2.metric("52W Position", f"{format_number(pos52, 1)} / 100")
        sub3.metric("1Y Z-score", format_number(z1y, 2))
        sub4.metric("Drawdown", format_pct(ddv, 2))

        helper_rows = []
        for helper in ["US Dollar Index", "S&P 500", "KOSPI", "WTI", "Gold"]:
            if helper in close_df.columns:
                corr = close_df[[fx_focus, helper]].dropna().pct_change().dropna().corr().iloc[0, 1]
                helper_rows.append({"Driver": helper, "20D Return Correlation": corr})
        helper_df = pd.DataFrame(helper_rows)
        if not helper_df.empty:
            st.dataframe(
                helper_df.style.format({"20D Return Correlation": "{:.2f}"}),
                use_container_width=True
            )

# ============================================================
# Rates tab
# ============================================================
with tab_rates:
    st.subheader("Rates & Yield Curve")

    if fred_data:
        col1, col2 = st.columns([1.2, 1.0])

        with col1:
            curve_df = build_yield_curve_df(fred_data)
            if not curve_df["Yield"].dropna().empty:
                st.plotly_chart(
                    make_yield_curve_chart(curve_df),
                    use_container_width=True,
                    key="rates_yield_curve"
                )
            else:
                st.info("Yield curve data unavailable.")

        with col2:
            us2y = get_series_last_valid(fred_data.get("US 2Y", pd.Series(dtype=float)))
            us10y_v = get_series_last_valid(fred_data.get("US 10Y", pd.Series(dtype=float)))
            us30y_v = get_series_last_valid(fred_data.get("US 30Y", pd.Series(dtype=float)))
            us3m_v = get_series_last_valid(fred_data.get("US 3M", pd.Series(dtype=float)))
            real10y = get_series_last_valid(fred_data.get("US Real 10Y", pd.Series(dtype=float)))

            spread_10_2 = us10y_v - us2y if not pd.isna(us10y_v) and not pd.isna(us2y) else np.nan
            spread_10_3m = us10y_v - us3m_v if not pd.isna(us10y_v) and not pd.isna(us3m_v) else np.nan

            r1, r2 = st.columns(2)
            r1.metric("US 2Y", format_pct(us2y, 2))
            r2.metric("US 10Y", format_pct(us10y_v, 2))

            r3, r4 = st.columns(2)
            r3.metric("10Y - 2Y", format_pct(spread_10_2, 2))
            r4.metric("10Y - 3M", format_pct(spread_10_3m, 2))

            r5, r6 = st.columns(2)
            r5.metric("US 30Y", format_pct(us30y_v, 2))
            r6.metric("Real 10Y", format_pct(real10y, 2))

            st.markdown("#### Quick Read")
            if not pd.isna(spread_10_2):
                if spread_10_2 < 0:
                    st.warning("Yield curve remains inverted or flat: growth concerns / restrictive policy signal.")
                else:
                    st.success("Yield curve is positively sloped: less recessionary than inversion regimes.")

        # Historical rates chart
        chart_candidates = ["US 2Y", "US 10Y", "US 30Y", "US Real 10Y", "Fed Funds Rate"]
        rate_df = pd.DataFrame({
            k: fred_data[k] for k in chart_candidates if k in fred_data and not fred_data[k].empty
        }).dropna(how="all")

        if not rate_df.empty:
            st.plotly_chart(
                make_line_chart(rate_df, "US Rates History", "Yield (%)"),
                use_container_width=True,
                key="rates_history"
            )

        # Rates vs TLT
        if "TLT" in close_df.columns and not rate_df.empty and "US 10Y" in rate_df.columns:
            aligned = pd.concat(
                [close_df["TLT"], rate_df["US 10Y"]],
                axis=1,
                join="inner"
            ).dropna()
            if not aligned.empty:
                st.plotly_chart(
                    make_dual_axis_chart(
                        aligned["TLT"],
                        aligned["US 10Y"],
                        "TLT",
                        "US 10Y Yield",
                        "TLT vs US 10Y"
                    ),
                    use_container_width=True,
                    key="rates_tlt_dual"
                )
    else:
        st.info("FRED API key not detected. Rates macro panel is limited.")
        fallback_assets = [a for a in ["TLT", "IEF", "TIP"] if a in norm_df.columns]
        if fallback_assets:
            st.plotly_chart(
                make_line_chart(norm_df[fallback_assets], "Bond ETF Relative Performance", "Index Level"),
                use_container_width=True,
                key="rates_fallback_bond_etf"
            )

# ============================================================
# Risk tab
# ============================================================
with tab_risk:
    st.subheader("Risk / Credit / Commodities")

    c1, c2, c3, c4 = st.columns(4)

    # VIX level
    c1.metric("VIX", format_number(vix_level, 2))
    c2.metric("USD/KRW 1M", format_pct(usdkrw_1m, 2))
    c3.metric("S&P 500 1M", format_pct(spx_1m, 2))
    c4.metric("WTI 1M", format_pct(wti_1m, 2))

    risk_assets = ["S&P 500", "Nasdaq 100", "KOSPI", "WTI", "Gold", "TLT", "TIP", "DBC", "HYG", "LQD", "US Dollar Index"]
    risk_assets = [a for a in risk_assets if a in norm_df.columns]
    if risk_assets:
        st.plotly_chart(
            make_line_chart(norm_df[risk_assets], "Cross-Asset Risk Monitoring (Base = 100)", "Index Level"),
            use_container_width=True,
            key="risk_cross_asset"
        )

    left, right = st.columns([1.2, 1.0])

    with left:
        if fred_data:
            credit_names = ["US HY OAS", "US IG OAS", "TED Spread"]
            credit_df = pd.DataFrame({
                k: fred_data[k] for k in credit_names if k in fred_data and not fred_data[k].empty
            }).dropna(how="all")

            if not credit_df.empty:
                st.plotly_chart(
                    make_line_chart(credit_df, "Credit Stress Indicators", "Spread (%) or Level"),
                    use_container_width=True,
                    key="risk_credit_stress"
                )
            else:
                st.info("Credit spread data unavailable from FRED.")
        else:
            hyg_lqd_assets = [a for a in ["HYG", "LQD"] if a in close_df.columns]
            if hyg_lqd_assets:
                st.plotly_chart(
                    make_line_chart(norm_df[hyg_lqd_assets], "Credit ETF Relative Performance", "Index Level"),
                    use_container_width=True,
                    key="risk_credit_etf"
                )

    with right:
        stress_rows = []

        stress_rows.append({
            "Indicator": "VIX",
            "Value": vix_level,
            "Status": get_status_text(vix_level, warn=18, danger=25),
        })

        stress_rows.append({
            "Indicator": "USD/KRW 1M Change",
            "Value": usdkrw_1m,
            "Status": get_status_text(usdkrw_1m, warn=1.0, danger=3.0),
        })

        stress_rows.append({
            "Indicator": "S&P 500 1M Change",
            "Value": spx_1m,
            "Status": get_status_text(spx_1m, warn=-2.0, danger=-5.0, reverse=True),
        })

        stress_rows.append({
            "Indicator": "WTI 1M Change",
            "Value": wti_1m,
            "Status": get_status_text(wti_1m, warn=5.0, danger=10.0),
        })

        if fred_data:
            hy_oas = get_series_last_valid(fred_data.get("US HY OAS", pd.Series(dtype=float)))
            ig_oas = get_series_last_valid(fred_data.get("US IG OAS", pd.Series(dtype=float)))
            stress_rows.append({
                "Indicator": "US HY OAS",
                "Value": hy_oas,
                "Status": get_status_text(hy_oas, warn=4.5, danger=6.0),
            })
            stress_rows.append({
                "Indicator": "US IG OAS",
                "Value": ig_oas,
                "Status": get_status_text(ig_oas, warn=1.5, danger=2.0),
            })

        stress_df = pd.DataFrame(stress_rows)
        st.dataframe(
            stress_df.style.format({"Value": "{:.2f}"}),
            use_container_width=True
        )

# ============================================================
# Korea Focus tab
# ============================================================
with tab_korea:
    st.subheader("Korea Focus")

    korea_assets = ["KOSPI", "KOSDAQ", "USD/KRW", "EUR/KRW", "EWY", "SOX", "US Dollar Index", "WTI", "Copper"]
    korea_assets = [a for a in korea_assets if a in close_df.columns]

    if korea_assets:
        st.plotly_chart(
            make_line_chart(norm_df[korea_assets], "Korea-Focused Relative Performance (Base = 100)", "Index Level"),
            use_container_width=True,
            key="korea_relative_perf"
        )

    kc1, kc2 = st.columns([1.2, 1.0])

    with kc1:
        compare_options = [a for a in ["KOSPI", "USD/KRW", "SOX", "WTI", "Copper", "US Dollar Index"] if a in close_df.columns]
        selected_compare = st.multiselect(
            "Select Korea-related comparison assets",
            options=compare_options,
            default=[a for a in ["KOSPI", "USD/KRW", "SOX"] if a in compare_options],
            key="korea_compare_multiselect"
        )

        if selected_compare:
            st.plotly_chart(
                make_line_chart(norm_df[selected_compare], "Selected Korea Monitoring Assets", "Index Level"),
                use_container_width=True,
                key="korea_selected_assets"
            )

    with kc2:
        korea_rows = []
        for a in ["KOSPI", "KOSDAQ", "USD/KRW", "EUR/KRW", "SOX", "WTI", "Copper"]:
            if a in close_df.columns:
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
        korea_df = pd.DataFrame(korea_rows)
        if not korea_df.empty:
            st.dataframe(
                korea_df.style.format({
                    "Last": "{:,.2f}",
                    "1M %": "{:.2f}%",
                    "3M %": "{:.2f}%",
                    "1Y %": "{:.2f}%",
                    "Drawdown %": "{:.2f}%",
                }),
                use_container_width=True
            )

    st.markdown("### Korea Interpretation Notes")
    note1, note2, note3 = st.columns(3)

    kospi_1m = calc_change_windows(close_df["KOSPI"].dropna()).get("1m", np.nan) if "KOSPI" in close_df.columns else np.nan
    sox_1m = calc_change_windows(close_df["SOX"].dropna()).get("1m", np.nan) if "SOX" in close_df.columns else np.nan
    copper_1m = calc_change_windows(close_df["Copper"].dropna()).get("1m", np.nan) if "Copper" in close_df.columns else np.nan

    note1.metric("KOSPI 1M", format_pct(kospi_1m, 2))
    note2.metric("SOX 1M", format_pct(sox_1m, 2))
    note3.metric("Copper 1M", format_pct(copper_1m, 2))

    interp = []
    if not pd.isna(usdkrw_1m) and usdkrw_1m > 2:
        interp.append("- USD/KRW 상승 폭이 커서 원화 약세 압력이 강화된 상태입니다.")
    if not pd.isna(sox_1m) and sox_1m > 3:
        interp.append("- 반도체 업황 기대가 한국 수출/대형주 심리에 우호적일 수 있습니다.")
    if not pd.isna(wti_1m) and wti_1m > 8:
        interp.append("- 유가 급등은 한국처럼 에너지 수입 의존 경제에 부담이 될 수 있습니다.")
    if not pd.isna(copper_1m) and copper_1m < -5:
        interp.append("- 구리 약세는 글로벌 경기 둔화 신호로 해석될 수 있습니다.")
    if not interp:
        interp.append("- 현재 Korea panel 상에서는 뚜렷한 단일 스트레스 신호보다 혼합 신호가 보입니다.")

    for line in interp:
        st.write(line)

# ============================================================
# Data Table tab
# ============================================================
with tab_data:
    st.subheader("Raw Monitoring Tables")

    st.markdown("### Latest Snapshot")
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
        use_container_width=True
    )

    st.markdown("### Downloadable Price Data")
    price_export = close_df.copy()
    csv = price_export.to_csv().encode("utf-8")
    st.download_button(
        label="Download price history CSV",
        data=csv,
        file_name="global_market_overview_prices.csv",
        mime="text/csv",
        key="download_price_csv"
    )

    if fred_data:
        st.markdown("### FRED Series Latest Values")
        fred_rows = []
        for name, s in fred_data.items():
            if s is not None and not s.dropna().empty:
                fred_rows.append({
                    "Series": name,
                    "Last": s.dropna().iloc[-1],
                    "Date": s.dropna().index[-1].strftime("%Y-%m-%d"),
                })
        fred_df = pd.DataFrame(fred_rows)
        if not fred_df.empty:
            st.dataframe(
                fred_df.style.format({"Last": "{:,.4f}"}),
                use_container_width=True
            )

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Tip: Add a valid FRED API key to unlock richer rates, credit, and macro sections. "
    "Yahoo Finance symbols may occasionally vary by region/data availability."
)
