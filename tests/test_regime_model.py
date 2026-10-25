"""Unit tests for regime classification and transition modeling."""

import numpy as np
import pytest
from src.data_ingestion import generate_synthetic_regime_series
from src.indicators import compute_technical_features
from src.regime_model import MarketRegimeModel


@pytest.fixture
def feature_dataset():
    raw_df = generate_synthetic_regime_series(n_days=200, seed=42)
    return compute_technical_features(raw_df)


def test_model_fit_and_predict(feature_dataset) -> None:
    model = MarketRegimeModel(n_regimes=4, random_state=42)
    model.fit(feature_dataset)

    assert model.is_fitted
    preds, probs = model.predict(feature_dataset)

    assert len(preds) == len(feature_dataset)
    assert probs.shape == (len(feature_dataset), 4)

    # Check valid regime states
    unique_regimes = set(np.unique(preds))
    assert unique_regimes.issubset({0, 1, 2, 3})

    # Probability rows must sum to ~1.0
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)


def test_transition_matrix_stochastic_property(feature_dataset) -> None:
    model = MarketRegimeModel(n_regimes=4, random_state=42)
    model.fit(feature_dataset)

    t_matrix = model.transition_matrix
    assert t_matrix.shape == (4, 4)

    # Each row should sum to 1.0 (or 0 if an unvisited state exists) within rounding tolerance
    for row in t_matrix:
        row_sum = np.sum(row)
        assert np.isclose(row_sum, 1.0, atol=1e-3) or np.isclose(row_sum, 0.0, atol=1e-3)


def test_regime_summary_computation(feature_dataset) -> None:
    model = MarketRegimeModel(n_regimes=4, random_state=42)
    model.fit(feature_dataset)
    preds, _ = model.predict(feature_dataset)
    summary = model.get_regime_summary(feature_dataset, preds)

    assert len(summary) == 4
    total_percentage = sum(r["percentage"] for r in summary)
    assert np.isclose(total_percentage, 100.0, atol=0.5)
