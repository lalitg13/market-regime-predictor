"""Unit tests for regime-conditioned backtesting engine."""

import numpy as np
import pytest
from src.backtester import RegimeBacktester
from src.data_ingestion import generate_synthetic_regime_series
from src.indicators import compute_technical_features


def test_regime_backtester_execution() -> None:
    raw_df = generate_synthetic_regime_series(n_days=150, seed=99)
    features_df = compute_technical_features(raw_df)

    # Alternate regimes
    regimes = np.array([i % 4 for i in range(len(features_df))])

    backtester = RegimeBacktester(initial_capital=100_000.0)
    result = backtester.run(features_df, regimes)

    assert "strategy_metrics" in result
    assert "benchmark_metrics" in result
    assert "equity_timeseries" in result

    sm = result["strategy_metrics"]
    bm = result["benchmark_metrics"]

    assert "sharpe_ratio" in sm
    assert "max_drawdown_pct" in sm
    assert "cagr_pct" in sm

    assert result["final_strategy_equity"] > 0
    assert result["final_benchmark_equity"] > 0
    assert len(result["equity_timeseries"]) > 0


def test_backtester_invalid_input_length() -> None:
    raw_df = generate_synthetic_regime_series(n_days=50, seed=1)
    backtester = RegimeBacktester()
    with pytest.raises(ValueError):
        backtester.run(raw_df, np.array([0, 1]))
