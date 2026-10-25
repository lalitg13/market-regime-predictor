"""Unit tests for quantitative technical indicators."""

import numpy as np
import pandas as pd
import pytest
from src.data_ingestion import generate_synthetic_regime_series
from src.indicators import compute_technical_features


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    return generate_synthetic_regime_series(n_days=100, initial_price=100.0, seed=123)


def test_compute_technical_features_structure(sample_market_data: pd.DataFrame) -> None:
    features_df = compute_technical_features(sample_market_data)

    expected_columns = [
        "Log_Return",
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "Trend_Ratio_50_200",
        "EMA_12",
        "EMA_26",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "RSI_14",
        "ATR_14",
        "ATR_Pct",
        "BB_Upper",
        "BB_Lower",
        "BB_Width",
        "BB_PctB",
        "Rolling_Vol_20",
        "Momentum_10",
        "Momentum_30",
        "Volume_Ratio",
    ]

    for col in expected_columns:
        assert col in features_df.columns, f"Expected column {col} missing in features."

    assert not features_df.isnull().values.any(), "Features DataFrame contains NaN values."
    assert not np.isinf(features_df[expected_columns].values).any(), "Features contain infinite values."


def test_rsi_bounds(sample_market_data: pd.DataFrame) -> None:
    features_df = compute_technical_features(sample_market_data)
    rsi = features_df["RSI_14"]
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all(), "RSI values exceed [0, 100] boundary."


def test_bollinger_bands_ordering(sample_market_data: pd.DataFrame) -> None:
    features_df = compute_technical_features(sample_market_data)
    upper = features_df["BB_Upper"].values
    lower = features_df["BB_Lower"].values
    assert (upper >= lower).all(), "BB Upper band must be greater than or equal to BB Lower band."
