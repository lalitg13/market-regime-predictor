"""FastAPI application for Quantitative Stock & Market Regime Predictor.

Provides high-performance REST API endpoints for market data ingestion,
technical indicator derivation, latent regime detection, and strategy backtesting.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.backtester import RegimeBacktester
from src.data_ingestion import fetch_market_data
from src.indicators import compute_technical_features
from src.regime_model import REGIME_COLORS, REGIME_NAMES, MarketRegimeModel

app = FastAPI(
    title="Market Regime Predictor API",
    description="Quantitative pipeline and machine learning engine for financial market regime classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


class AnalysisRequest(BaseModel):
    ticker: str = Field(default="SPY", description="Market ticker symbol (e.g. SPY, QQQ, AAPL, NVDA)")
    period: str = Field(default="2y", description="Lookback window (e.g. 1y, 2y, 5y)")
    n_regimes: int = Field(default=4, ge=2, le=6, description="Number of discrete market regimes to classify")


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "HEALTHY", "system": "MarketRegimePredictor", "version": "1.0.0"}


@app.post("/api/analyze")
def analyze_market_regimes(req: AnalysisRequest) -> Dict[str, Any]:
    """Execute complete end-to-end quantitative regime prediction and backtesting pipeline."""
    try:
        raw_df = fetch_market_data(ticker=req.ticker, period=req.period)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch market data: {str(exc)}")

    features_df = compute_technical_features(raw_df)

    # Train regime classification model
    model = MarketRegimeModel(n_regimes=req.n_regimes)
    model.fit(features_df)
    predicted_regimes, probabilities = model.predict(features_df)

    # Backtest regime allocation strategy
    backtester = RegimeBacktester()
    backtest_results = backtester.run(features_df, predicted_regimes)

    # Regime summary metrics
    summary = model.get_regime_summary(features_df, predicted_regimes)

    # Current regime status
    latest_idx = -1
    latest_regime_id = int(predicted_regimes[latest_idx])
    latest_probs = [round(float(p), 4) for p in probabilities[latest_idx]]

    latest_row = features_df.iloc[-1]
    prev_row = features_df.iloc[-2] if len(features_df) > 1 else latest_row
    price_change = float(latest_row["Close"] - prev_row["Close"])
    price_change_pct = float((price_change / prev_row["Close"]) * 100) if prev_row["Close"] != 0 else 0.0

    # Downsampled price history for charts
    n_total = len(features_df)
    step = max(1, n_total // 250)
    sampled_indices = list(range(0, n_total, step))
    if (n_total - 1) not in sampled_indices:
        sampled_indices.append(n_total - 1)

    chart_series = []
    for idx in sampled_indices:
        row = features_df.iloc[idx]
        reg_id = int(predicted_regimes[idx])
        chart_series.append(
            {
                "date": str(row["Date"]),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
                "regime_id": reg_id,
                "regime_name": REGIME_NAMES.get(reg_id, "Unknown"),
                "regime_color": REGIME_COLORS.get(reg_id, "#888888"),
                "sma_50": round(float(row["SMA_50"]), 2),
                "sma_200": round(float(row["SMA_200"]), 2),
                "rsi_14": round(float(row["RSI_14"]), 2),
            }
        )

    return {
        "ticker": req.ticker.upper(),
        "period": req.period,
        "sample_count": n_total,
        "current_regime": {
            "regime_id": latest_regime_id,
            "name": REGIME_NAMES.get(latest_regime_id, "Unknown"),
            "color": REGIME_COLORS.get(latest_regime_id, "#888888"),
            "confidence_pct": round(float(latest_probs[latest_regime_id]) * 100, 2),
            "probabilities": latest_probs,
        },
        "latest_telemetry": {
            "close": round(float(latest_row["Close"]), 2),
            "change": round(price_change, 2),
            "change_pct": round(price_change_pct, 2),
            "rsi_14": round(float(latest_row["RSI_14"]), 2),
            "macd_hist": round(float(latest_row["MACD_Hist"]), 3),
            "rolling_vol_20": round(float(latest_row["Rolling_Vol_20"]) * 100, 2),
            "atr_14": round(float(latest_row["ATR_14"]), 2),
            "bb_pct_b": round(float(latest_row["BB_PctB"]) * 100, 2),
        },
        "regime_summary": summary,
        "transition_matrix": model.transition_matrix.tolist(),
        "feature_importances": model.feature_importances,
        "backtest": backtest_results,
        "price_history": chart_series,
    }


if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend asset not found")
