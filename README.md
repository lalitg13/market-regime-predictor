# Market Regime Predictor

[![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.15-3776ab.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-f7931e.svg?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Status: Production](https://img.shields.io/badge/Status-Production--Grade-00ff66.svg?style=flat-square)](#)

> [!NOTE]
> A quantitative tool that detects market states (bull, bear, volatile, consolidating) using machine learning and backtests a dynamic trading strategy in real time.

---

## Overview

Market Regime Predictor downloads historical asset prices, computes technical indicators, and classifies market conditions using Gaussian Mixture clustering and Random Forest models. It includes a Bloomberg-terminal-styled dark dashboard for viewing live price regimes and backtest results.

```
+-----------------------------------------------------------------------+
| Market Data  -->  Technical Features  -->  Machine Learning  -->  Web |
| (Yahoo / Mock)    (RSI, MACD, Vol)         (Regime Detection)     (UI)|
+-----------------------------------------------------------------------+
```

---

## Key Features

- **Four Market Regimes**: Automatically classifies periods into Bull Trending, Bear Trending, High Volatility, and Low Volatility.
- **Built-in Backtester**: Simulates an asset allocation strategy that adapts to detected regimes and compares returns against buy-and-hold.
- **Financial Terminal Interface**: Monospace dark-mode dashboard with interactive charts, metrics cards, and live logs.
- **Offline Fallback**: Automatically generates realistic synthetic price series if market data feeds are unavailable.

---

## Market Regimes Explained

| Regime | Color | Description | Strategy Action |
|---|---|---|---|
| **Bull Trending** | Green | Steady upward price action, moderate volatility | 100% Long Equity |
| **Bear Trending** | Red | Downward price action, elevated volatility | 100% Cash (Capital Preservation) |
| **High Volatility** | Amber | Choppy price swings, wide Bollinger bands | 25% Defensive Allocation |
| **Consolidation** | Blue | Sideways price movement, low volatility | 60% Moderate Allocation |

---

## Quickstart

### 1. Installation

```bash
cd ~/Desktop/Workspace/MarketRegimePredictor
pip install -r requirements.txt
```

### 2. Run the Application

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to: **http://localhost:8000**

### 3. Run Tests

```bash
pytest -v tests/
```

---

## API Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check and system status |
| `POST` | `/api/analyze` | Ingests ticker data, classifies regimes, and returns backtest results |

**Example Request:**
```json
{
  "ticker": "SPY",
  "period": "2y",
  "n_regimes": 4
}
```

---

## License

Released under the MIT License.
