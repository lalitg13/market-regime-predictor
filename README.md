# Market Regime Predictor

[![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.15-3776ab.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-f7931e.svg?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Build: Production](https://img.shields.io/badge/Status-Production--Grade-00ff66.svg?style=flat-square)](#)

> [!NOTE]
> Quantitative asset allocation and market regime classification pipeline utilizing unsupervised Gaussian Mixture latent space clustering and supervised Random Forest calibration. Integrates dynamic Markov transition matrix derivation with historical strategy backtesting.

---

## Technical Architecture

```
+-------------------------------------------------------------------------------------------------+
|                                 MARKET REGIME PREDICTOR PIPELINE                                |
+-------------------------------------------------------------------------------------------------+
|                                                                                                 |
|   +-----------------------+      +-----------------------+      +---------------------------+   |
|   |  Yahoo Finance API    | ---> |  Data Ingestion       | ---> |  Feature Engineering      |   |
|   |  or Synthetic Engine  |      |  (OHLCV Normalization)|      |  (8 Vol / Trend / Mom)    |   |
|   +-----------------------+      +-----------------------+      +---------------------------+   |
|                                                                               |                 |
|                                                                               v                 |
|   +-----------------------+      +-----------------------+      +---------------------------+   |
|   |  Dynamic Strategy     | <--- |  Random Forest        | <--- |  Gaussian Mixture         |   |
|   |  Backtester           |      |  Supervised Ensemble  |      |  Unsupervised Clustering  |   |
|   +-----------------------+      +-----------------------+      +---------------------------+   |
|               |                              |                                                  |
|               v                              v                                                  |
|   +-----------------------------------------------------------------------------------------+   |
|   |  FastAPI Backend (REST API Endpoints: /api/analyze, /api/health)                        |   |
|   +-----------------------------------------------------------------------------------------+   |
|                                              |                                                  |
|                                              v                                                  |
|   +-----------------------------------------------------------------------------------------+   |
|   |  Dense Financial Terminal UI (JetBrains Mono, Canvas Time-Series, Hex #0d1117 Theme)    |   |
|   +-----------------------------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------------------------+
```

---

## Market Regime Specification

The system partitions asset returns into discrete macroeconomic states:

| Regime ID | Semantic Designation | Telemetry Color | Mathematical Profile | Target Allocation Weight |
|---|---|---|---|---|
| **State 0** | Bull Trending | `#00ff66` Emerald Green | Positive drift, moderate volatility, RSI > 50 | 100% Long Equity |
| **State 1** | Bear Trending | `#ff3344` Crimson Red | Negative drift, elevated volatility, RSI < 45 | 0% Cash Preservation |
| **State 2** | High Volatility Mean-Reverting | `#ffaa00` Amber Warning | Elevated ATR, wide Bollinger Bands, extreme swings | 25% Defensive / Hedged |
| **State 3** | Low Volatility Consolidation | `#00b4d8` Ice Blue | Low ATR, compressed BB width, sideways return | 60% Moderate Equity |

---

## Mathematical Formulation

### 1. Latent State Inference
Let $\mathbf{x}_t \in \mathbb{R}^d$ represent the feature vector at time $t$, normalized via standard score transformation:

$$\mathbf{z}_t = \mathbf{\Sigma}^{-1/2} (\mathbf{x}_t - \mathbf{\mu})$$

The probability density of observing $\mathbf{z}_t$ under a $K$-component Gaussian Mixture Model is:

$$p(\mathbf{z}_t) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{z}_t \mid \mathbf{\mu}_k, \mathbf{\Sigma}_k)$$

Where $\pi_k$ represents the prior mixing weight of regime $k$, and $\sum_{k=1}^K \pi_k = 1$.

### 2. Empirical Markov Transition Matrix
The state transition probability $P_{ij}$ from regime $i$ at time $t$ to regime $j$ at time $t+1$ is computed empirically:

$$P_{ij} = \mathbb{P}(S_{t+1} = j \mid S_t = i) = \frac{N_{ij}}{\sum_{k=1}^K N_{ik}}$$

Where $N_{ij}$ is the count of historical transitions observed from state $i$ to state $j$.

### 3. Quantitative Risk Metrics
Portfolio performance is evaluated against the buy-and-hold benchmark:

- **Sharpe Ratio:**
  $$\text{SR} = \frac{\mathbb{E}[R_p - R_f]}{\sigma_p} \cdot \sqrt{252}$$

- **Sortino Ratio:**
  $$\text{SoR} = \frac{\mathbb{E}[R_p - R_f]}{\sqrt{\mathbb{E}[\min(0, R_p - R_f)^2]}} \cdot \sqrt{252}$$

- **Maximum Drawdown:**
  $$\text{MDD} = \max_{\tau \in [0, t]} \left( \frac{\max_{s \le \tau} V_s - V_\tau}{\max_{s \le \tau} V_s} \right)$$

---

## Feature Matrix Definition

| Feature | Computation | Economic Rationale |
|---|---|---|
| `Log_Return` | $\ln(P_t / P_{t-1})$ | Continuous rate of return |
| `SMA_20`, `SMA_50`, `SMA_200` | Rolling mean over windows 20, 50, 200 | Moving trend baseline |
| `Trend_Ratio_50_200` | $\text{SMA}_{50} / \text{SMA}_{200}$ | Golden cross / Death cross indicator |
| `MACD_Hist` | $(\text{EMA}_{12} - \text{EMA}_{26}) - \text{Signal}_9$ | Second-derivative momentum oscillator |
| `RSI_14` | $100 - (100 / (1 + \text{RS}))$ | Mean-reversion oscillator boundary |
| `ATR_Pct` | $\text{ATR}_{14} / P_t$ | Normalized absolute price dispersion |
| `BB_Width` | $(U_{20} - L_{20}) / \text{MA}_{20}$ | Volatility contraction / expansion squeeze |
| `Rolling_Vol_20` | $\text{Std}(\text{Log\_Return}_{20}) \cdot \sqrt{252}$ | Annualized short-term historical volatility |

---

## REST API Specification

### Endpoint: `POST /api/analyze`
Executes ingestion, feature generation, regime classification, and portfolio backtest.

**Request Payload:**
```json
{
  "ticker": "SPY",
  "period": "2y",
  "n_regimes": 4
}
```

**Response Payload Sample:**
```json
{
  "ticker": "SPY",
  "period": "2y",
  "sample_count": 504,
  "current_regime": {
    "regime_id": 0,
    "name": "Bull Trending",
    "color": "#00ff66",
    "confidence_pct": 89.42,
    "probabilities": [0.8942, 0.0210, 0.0418, 0.0430]
  },
  "latest_telemetry": {
    "close": 582.40,
    "change": 3.12,
    "change_pct": 0.54,
    "rsi_14": 62.4,
    "macd_hist": 0.42,
    "rolling_vol_20": 11.82,
    "atr_14": 5.10,
    "bb_pct_b": 78.4
  },
  "backtest": {
    "initial_capital": 100000.0,
    "final_strategy_equity": 142850.20,
    "strategy_metrics": {
      "cagr_pct": 19.45,
      "sharpe_ratio": 1.62,
      "sortino_ratio": 2.14,
      "max_drawdown_pct": -8.12,
      "win_rate_pct": 58.20
    }
  }
}
```

---

## Installation & Execution

### 1. Requirements
Ensure Python 3.9+ is installed:
```bash
python3 --version
```

### 2. Dependency Installation
```bash
pip install -r requirements.txt
```

### 3. Run Test Suite
```bash
pytest -v tests/
```

### 4. Launch Production Server
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
Access the terminal interface at `http://localhost:8000`.

---

## License

This software is released under the MIT License. Refer to the LICENSE file for details.
