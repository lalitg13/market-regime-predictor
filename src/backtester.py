"""Regime-conditioned algorithmic backtester.

Executes tactical asset allocation conditioned on predicted market states,
comparing risk-adjusted performance against a standard buy-and-hold benchmark.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd


class RegimeBacktester:
    """Simulates performance of dynamic regime allocation strategy."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.02,
        transaction_cost_bps: float = 5.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost_bps / 10_000.0

        # Allocation weights conditioned on regime state:
        # 0: Bull Trending -> 100% Long
        # 1: Bear Trending -> 0% (100% Cash preserve capital)
        # 2: High Volatility -> 25% Defensive allocation
        # 3: Low Volatility -> 60% Moderate exposure
        self.allocation_weights: Dict[int, float] = {
            0: 1.00,
            1: 0.00,
            2: 0.25,
            3: 0.60,
        }

    def run(self, df: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Execute backtest over historical price and regime sequences.

        Args:
            df: Price dataframe with Date and Close columns.
            regimes: Array of predicted market regime indices.

        Returns:
            Dictionary containing performance metrics and timeseries data.
        """
        n = len(df)
        if n < 2 or len(regimes) != n:
            raise ValueError("Input data and regime array length mismatch.")

        dates = df["Date"].tolist()
        close_prices = df["Close"].values.astype(float)
        daily_returns = np.zeros(n)
        daily_returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]

        # Calculate allocation weights shifted by 1 to eliminate lookahead bias
        raw_weights = np.array([self.allocation_weights.get(int(r), 0.5) for r in regimes])
        shifted_weights = np.zeros(n)
        shifted_weights[1:] = raw_weights[:-1]  # Trade at close t based on regime t-1

        # Transaction cost deduction upon weight reallocation
        turnover = np.abs(np.diff(shifted_weights, prepend=0.0))
        tc_impact = turnover * self.transaction_cost

        # Strategy returns
        strat_daily_returns = (shifted_weights * daily_returns) - tc_impact

        # Equity curves
        benchmark_curve = self.initial_capital * np.cumprod(1.0 + daily_returns)
        strategy_curve = self.initial_capital * np.cumprod(1.0 + strat_daily_returns)

        # Performance analytics
        bench_metrics = self._compute_performance_metrics(daily_returns, benchmark_curve)
        strat_metrics = self._compute_performance_metrics(strat_daily_returns, strategy_curve)

        timeseries: List[Dict[str, Any]] = []
        step = max(1, n // 150)
        for i in range(0, n, step):
            timeseries.append(
                {
                    "date": dates[i],
                    "close": round(float(close_prices[i]), 2),
                    "benchmark_equity": round(float(benchmark_curve[i]), 2),
                    "strategy_equity": round(float(strategy_curve[i]), 2),
                    "regime": int(regimes[i]),
                    "allocation": round(float(shifted_weights[i]), 2),
                }
            )

        return {
            "initial_capital": self.initial_capital,
            "final_benchmark_equity": round(float(benchmark_curve[-1]), 2),
            "final_strategy_equity": round(float(strategy_curve[-1]), 2),
            "strategy_metrics": strat_metrics,
            "benchmark_metrics": bench_metrics,
            "equity_timeseries": timeseries,
        }

    def _compute_performance_metrics(
        self, returns: np.ndarray, equity_curve: np.ndarray
    ) -> Dict[str, float]:
        """Compute key portfolio risk and performance ratios."""
        n_days = len(returns)
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        cagr = (1.0 + total_return) ** (252.0 / max(1, n_days)) - 1.0

        daily_rf = self.risk_free_rate / 252.0
        excess_returns = returns - daily_rf
        ann_vol = float(np.std(returns) * np.sqrt(252))

        # Sharpe Ratio
        sharpe = float(np.mean(excess_returns) / (np.std(returns) + 1e-8) * np.sqrt(252))

        # Sortino Ratio (downside risk only)
        downside_returns = returns[returns < 0]
        downside_dev = float(np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 0 else 1e-6
        sortino = float(np.mean(excess_returns) * np.sqrt(252) / (downside_dev + 1e-8))

        # Maximum Drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = float(np.min(drawdowns))

        # Win rate
        positive_days = np.sum(returns > 0)
        traded_days = np.sum(returns != 0)
        win_rate = float(positive_days / max(1, traded_days))

        return {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "annualized_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
        }
