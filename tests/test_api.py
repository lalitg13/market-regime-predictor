"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["system"] == "MarketRegimePredictor"


def test_api_analyze_endpoint():
    payload = {"ticker": "SPY", "period": "1y", "n_regimes": 4}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["ticker"] == "SPY"
    assert "current_regime" in data
    assert "latest_telemetry" in data
    assert "regime_summary" in data
    assert "transition_matrix" in data
    assert "feature_importances" in data
    assert "backtest" in data
    assert "price_history" in data
    assert len(data["price_history"]) > 0
