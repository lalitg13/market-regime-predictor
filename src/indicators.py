"""Technical indicators and quantitative feature engineering pipeline.

Computes mathematically sound momentum, trend, volatility, and volume indicators
from raw price and volume series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate quantitative technical features from standard OHLCV dataframe.

    Features generated:
        - Log Returns: Daily logarithmic returns
        - SMA_20, SMA_50, SMA_200: Trend moving averages
        - Trend_Ratio_50_200: Ratio of 50-day to 200-day SMA (Golden/Death cross signal)
        - EMA_12, EMA_26, MACD, MACD_Signal, MACD_Hist: Trend momentum oscillator
        - RSI_14: Relative Strength Index (0 to 100)
        - ATR_14: Normalized Average True Range (volatility proxy)
        - BB_Upper, BB_Lower, BB_Width, BB_PctB: Bollinger bands and squeeze metrics
        - Rolling_Vol_20: Annualized 20-day historical standard deviation
        - Momentum_10, Momentum_30: Price rate of change
        - Volume_Ratio: Current volume relative to 20-day SMA volume

    Args:
        df: Input DataFrame containing Open, High, Low, Close, Volume.

    Returns:
        Augmented DataFrame with NaN values cleaned via backfill/forward-fill.
    """
    data = df.copy()
    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float)

    # Log Returns
    data["Log_Return"] = np.log(close / close.shift(1)).fillna(0.0)

    # Trend moving averages
    data["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    data["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    data["SMA_200"] = close.rolling(window=200, min_periods=1).mean()
    data["Trend_Ratio_50_200"] = data["SMA_50"] / np.where(data["SMA_200"] == 0, 1e-6, data["SMA_200"])

    # Exponential moving averages and MACD
    data["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # Relative Strength Index (RSI)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = np.where(avg_loss == 0, 100.0, avg_gain / np.where(avg_loss == 0, 1e-6, avg_loss))
    data["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))
    data["RSI_14"] = data["RSI_14"].clip(0.0, 100.0).fillna(50.0)

    # Average True Range (ATR)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["ATR_14"] = tr.rolling(window=14, min_periods=1).mean()
    data["ATR_Pct"] = data["ATR_14"] / np.where(close == 0, 1e-6, close)

    # Bollinger Bands
    rolling_mean_20 = close.rolling(window=20, min_periods=1).mean()
    rolling_std_20 = close.rolling(window=20, min_periods=1).std().fillna(0.0)
    data["BB_Upper"] = rolling_mean_20 + (2.0 * rolling_std_20)
    data["BB_Lower"] = rolling_mean_20 - (2.0 * rolling_std_20)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / np.where(rolling_mean_20 == 0, 1e-6, rolling_mean_20)
    band_range = data["BB_Upper"] - data["BB_Lower"]
    data["BB_PctB"] = np.where(band_range == 0, 0.5, (close - data["BB_Lower"]) / np.where(band_range == 0, 1e-6, band_range))

    # Rolling Volatility (20-day annualized, sqrt(252))
    data["Rolling_Vol_20"] = data["Log_Return"].rolling(window=20, min_periods=2).std().fillna(0.0) * np.sqrt(252)

    # Momentum
    data["Momentum_10"] = (close / close.shift(10).replace(0, np.nan) - 1.0).fillna(0.0)
    data["Momentum_30"] = (close / close.shift(30).replace(0, np.nan) - 1.0).fillna(0.0)

    # Volume Ratio
    vol_sma_20 = volume.rolling(window=20, min_periods=1).mean()
    data["Volume_Ratio"] = volume / np.where(vol_sma_20 == 0, 1.0, vol_sma_20)

    # Clean extreme infinities or NaNs
    data = data.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0.0)

    return data
