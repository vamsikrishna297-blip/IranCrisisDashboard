import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import re

st.set_page_config(layout="wide")

# -------------------------------
# CONFIG
# -------------------------------
TICKER_FILE = r"C:\TradingStrategies\NarrativeIntensity\production\Data\StockPrices\nifty500_tickers.txt"
START_DATE = "2026-01-01"

LOOKBACK_MONTHS = 10
SKIP_MONTHS = 1
TOP_N = 10
BOTTOM_N = 10

OIL_TICKER = "BZ=F"
VIX_TICKER = "^INDIAVIX"

# -------------------------------
# LOAD & CLEAN UNIVERSE
# -------------------------------
def load_universe(file_path):
    import re

    with open(file_path, "r") as f:
        raw = f.read()

    tickers = re.split(r"[,\n]+", raw)
    tickers = [t.strip() for t in tickers if t.strip() != ""]

    # Keep only valid NSE tickers
    tickers = [t for t in tickers if t.endswith(".NS")]

    # Remove problematic patterns
    invalid_patterns = ["&", ".BE"]
    tickers = [t for t in tickers if not any(p in t for p in invalid_patterns)]

    # Fix known ticker issues
    fixes = {
        "L&TTS.NS": "LTTS.NS",
        "GMRINFRASTRUCT.NS": "GMRINFRA.NS"
    }

    tickers = [fixes.get(t, t) for t in tickers]

    return list(set(tickers))

universe = load_universe(TICKER_FILE)

st.write(f"Universe Size: {len(universe)}")

# -------------------------------
# FETCH DATA
# -------------------------------
@st.cache_data
def fetch_prices(tickers):
    data = yf.download(tickers, start=START_DATE, progress=False, group_by='ticker')

    if data.empty:
        return pd.DataFrame()

    # Case 1: Multiple tickers → MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[1]:
            data = data.xs("Adj Close", axis=1, level=1)
        else:
            data = data.xs("Close", axis=1, level=1)

    # Case 2: Single ticker
    else:
        if "Adj Close" in data.columns:
            data = data[["Adj Close"]]
        elif "Close" in data.columns:
            data = data[["Close"]]
        else:
            return pd.DataFrame()

    # Drop columns with all NaNs (failed tickers)
    data = data.dropna(axis=1, how="all")

    return data

oil = fetch_prices([OIL_TICKER])
vix = fetch_prices([VIX_TICKER])

import os

def load_parquet_data(folder_path):
    price_df = None

    for file in os.listdir(folder_path):
        if file.endswith(".parquet") and "index" not in file.lower():

            df = pd.read_parquet(os.path.join(folder_path, file))

            if df.empty:
                continue

            # ---- CLEAN COLUMN NAMES (same as your working script) ----
            df.columns = [
                re.sub(r"[(')\s]", "", col).split(",")[0] if isinstance(col, str) else col
                for col in df.columns
            ]
            df.columns = [col.lower() for col in df.columns]

            # ---- REQUIRED COLUMNS ----
            if "date" not in df.columns or "close" not in df.columns:
                continue

            # ---- EXTRACT TICKER FROM FILE NAME ----
            stock = os.path.splitext(os.path.basename(file))[0].upper()

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            # Reduce memory (important)
            df = df[df["date"] >= "2023-01-01"]

            df.set_index("date", inplace=True)

            # Convert to single-column DataFrame
            df = df[["close"]].rename(columns={"close": stock})

            # Merge incrementally
            if price_df is None:
                price_df = df
            else:
                price_df = price_df.join(df, how="outer")

    if price_df is None:
        raise ValueError("No valid parquet files found")

    return price_df.sort_index()


# Use your existing folder
DATA_PATH = r"C:\TradingStrategies\NarrativeIntensity\python_scripts\GDELT\StockReturnsMarketBetaNifty500"

prices = load_parquet_data(DATA_PATH)


# Drop stocks with too many missing values
prices = prices.dropna(axis=1, thresh=int(0.7 * len(prices)))

# -------------------------------
# MOM801 (MONTHLY)
# -------------------------------
monthly_prices = prices.resample("ME").last()
monthly_returns = monthly_prices.pct_change()

mom = (
    (1 + monthly_returns.shift(SKIP_MONTHS))
    .rolling(LOOKBACK_MONTHS)
    .apply(np.prod, raw=True) - 1
)

# -------------------------------
# DAILY RETURNS
# -------------------------------
daily_returns = prices.pct_change()

# -------------------------------
# MONTHLY FREEZE -> DAILY TRACKING
# -------------------------------
# Use previous month-end MOM scores so each month's portfolio is frozen and
# then tracked day-by-day for that whole month.
signal_monthly = mom.shift(1)
signal_daily = signal_monthly.reindex(daily_returns.index, method="ffill")

top_port = pd.Series(index=daily_returns.index, dtype=float)
bottom_port = pd.Series(index=daily_returns.index, dtype=float)

for dt in daily_returns.index:
    scores = signal_daily.loc[dt].dropna()
    if len(scores) < max(TOP_N, BOTTOM_N):
        continue

    top_names = scores.nlargest(TOP_N).index
    bottom_names = scores.nsmallest(BOTTOM_N).index

    top_port.loc[dt] = daily_returns.loc[dt, top_names].mean()
    bottom_port.loc[dt] = daily_returns.loc[dt, bottom_names].mean()

# Latest month portfolio for display
latest_signal = signal_daily.dropna(how="all").iloc[-1].dropna()
top_stocks = latest_signal.nlargest(TOP_N).index.tolist()
bottom_stocks = latest_signal.nsmallest(BOTTOM_N).index.tolist()

# -------------------------------
# MOMENTUM HEALTH
# -------------------------------
momentum_spread = top_port - bottom_port
momentum_health = momentum_spread.rolling(3).mean()

# -------------------------------
# OIL & VIX SIGNALS
# -------------------------------
oil_ret = oil.pct_change(3)
vix_ret = vix.pct_change(3)

def oil_score(x):
    if x < -0.02:
        return 1
    elif x > 0.02:
        return -1
    return 0

def vol_score(x):
    if x < -0.05:
        return 1
    elif x > 0.05:
        return -1
    return 0

def mom_score(x):
    if x > 0:
        return 1
    elif x < 0:
        return -2
    return 0

oil_signal = oil_ret.squeeze().apply(oil_score)
vol_signal = vix_ret.squeeze().apply(vol_score)
mom_signal = momentum_health.apply(mom_score)

# -------------------------------
# DASHBOARD
# -------------------------------
dashboard = pd.DataFrame(index=momentum_health.index)

dashboard["Momentum"] = mom_signal
dashboard["Oil"] = oil_signal
dashboard["Vol"] = vol_signal

dashboard["Total"] = dashboard.sum(axis=1)

def regime(score):
    if score >= 2:
        return "STRONG"
    elif score == 1:
        return "OK"
    elif score == 0:
        return "FRAGILE"
    else:
        return "EXIT"

dashboard["Regime"] = dashboard["Total"].apply(regime)

latest = dashboard.iloc[-1]

# -------------------------------
# UI
# -------------------------------
st.title("📊 Momentum Dashboard (MOM801 - Nifty 500)")

st.write("### Top 10 Momentum Stocks")
st.write(top_stocks)

st.write("### Bottom 10 (Reference)")
st.write(bottom_stocks)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Momentum", latest["Momentum"])
col2.metric("Oil", latest["Oil"])
col3.metric("Vol", latest["Vol"])
col4.metric("Total Score", latest["Total"])

# -------------------------------
# REGIME MESSAGE
# -------------------------------
if latest["Regime"] == "STRONG":
    st.success("STRONG → Stay Aggressive")
elif latest["Regime"] == "OK":
    st.info("OK → Hold")
elif latest["Regime"] == "FRAGILE":
    st.warning("FRAGILE → Reduce")
else:
    st.error("EXIT → Cut Exposure")

# -------------------------------
# PLOTS
# -------------------------------
st.subheader("Momentum Spread")
st.line_chart(momentum_spread)

st.subheader("Oil (3D Return)")
st.line_chart(oil_ret)

st.subheader("VIX (3D Return)")
st.line_chart(vix_ret)

st.subheader("Regime Score")
st.line_chart(dashboard["Total"])
