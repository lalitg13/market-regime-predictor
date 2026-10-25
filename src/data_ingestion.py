"""Market data ingestion module.

Provides resilient historical market data retrieval using Yahoo Finance,
with an algorithmic synthetic data generator fallback for offline operation,
testing, and continuous simulation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_regime_series(
    n_days: int = 500,
    initial_price: float = 150.0,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data using a multi-state Markov switching process.

    States:
        0: Bull Trending (positive drift, moderate volatility)
        1: Bear Trending (negative drift, elevated volatility)
        2: High Volatility Mean-Reverting (near-zero drift, spike volatility)
        3: Low Volatility Consolidation (tight range, low volume)

    Args:
        n_days: Number of trading days to simulate.
        initial_price: Starting equity price.
        seed: Random generator seed for reproducibility.

    Returns:
        DataFrame containing Date, Open, High, Low, Close, Volume, and GroundTruthRegime.
    """
    if seed is not None:
        np.random.seed(seed)

    # State parameters: (daily_drift, daily_volatility)
    regime_params = {
        0: (0.0008, 0.010),   # Bull
        1: (-0.0010, 0.018),  # Bear
        2: (0.0001, 0.025),   # High Vol
        3: (0.0002, 0.006),   # Low Vol
    }

    # Transition probability matrix (P_ij = prob of transitioning from i to j)
    p_matrix = np.array([
        [0.92, 0.03, 0.03, 0.02],
        [0.05, 0.88, 0.05, 0.02],
        [0.04, 0.06, 0.82, 0.08],
        [0.05, 0.03, 0.04, 0.88],
    ])

    current_state = 0
    states = [current_state]

    for _ in range(1, n_days):
        current_state = np.random.choice(4, p=p_matrix[current_state])
        states.append(current_state)

    returns = np.zeros(n_days)
    for i, state in enumerate(states):
        drift, vol = regime_params[state]
        returns[i] = np.random.normal(drift, vol)

    prices = initial_price * np.exp(np.cumsum(returns))

    start_date = datetime.now() - timedelta(days=int(n_days * 1.45))
    date_range = pd.bdate_range(start=start_date, periods=n_days)

    opens = np.zeros(n_days)
    highs = np.zeros(n_days)
    lows = np.zeros(n_days)
    closes = prices
    volumes = np.zeros(n_days, dtype=np.int64)

    for i in range(n_days):
        vol_scale = regime_params[states[i]][1]
        intraday_noise = np.random.normal(0, vol_scale * 0.5, 3)
        opens[i] = closes[i - 1] * (1 + intraday_noise[0]) if i > 0 else initial_price
        c = closes[i]
        o = opens[i]
        highs[i] = max(o, c) * (1 + abs(intraday_noise[1]))
        lows[i] = min(o, c) * (1 - abs(intraday_noise[2]))
        base_vol = 1_500_000
        vol_multiplier = 1.0 + (vol_scale / 0.010) * 0.8
        volumes[i] = int(base_vol * vol_multiplier * np.random.uniform(0.7, 1.4))

    df = pd.DataFrame(
        {
            "Date": date_range.strftime("%Y-%m-%d"),
            "Open": np.round(opens, 2),
            "High": np.round(highs, 2),
            "Low": np.round(lows, 2),
            "Close": np.round(closes, 2),
            "Volume": volumes,
            "GroundTruthRegime": states,
        }
    )
    return df


def fetch_market_data(
    ticker: str = "SPY",
    period: str = "2y",
    interval: str = "1d",
    use_fallback: bool = True,
) -> pd.DataFrame:
    """Fetch market price history for a given ticker or fall back to synthetic data.

    Args:
        ticker: Equity or index symbol (e.g. SPY, QQQ, AAPL, NVDA).
        period: Historical lookback period (e.g. 1y, 2y, 5y).
        interval: Data sampling interval.
        use_fallback: Whether to generate synthetic data if provider fails.

    Returns:
        Structured pandas DataFrame with standard market columns.
    """
    clean_ticker = ticker.strip().upper()
    try:
        import yfinance as yf

        data = yf.download(
            tickers=clean_ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if data is not None and not data.empty and len(data) > 30:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            data = data.reset_index()
            date_col = "Date" if "Date" in data.columns else data.columns[0]
            data["Date"] = pd.to_datetime(data[date_col]).dt.strftime("%Y-%m-%d")

            cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            available = [c for c in cols if c in data.columns]
            df = data[available].copy()
            for numeric_col in ["Open", "High", "Low", "Close"]:
                if numeric_col in df.columns:
                    df[numeric_col] = df[numeric_col].astype(float).round(2)
            if "Volume" in df.columns:
                df["Volume"] = df["Volume"].astype(int)
            return df
        else:
            logger.warning(
                "Upstream provider returned empty series for ticker: %s", clean_ticker
            )
    except Exception as exc:
        logger.warning(
            "Failed fetching %s from remote provider: %s. Initiating synthetic engine.",
            clean_ticker,
            exc,
        )

    if use_fallback:
        seed_hash = abs(hash(clean_ticker)) % (2**31)
        return generate_synthetic_regime_series(n_days=504, initial_price=200.0, seed=seed_hash)

    raise ValueError(f"Unable to retrieve data for ticker: {clean_ticker}")
