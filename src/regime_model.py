"""Machine learning regime classification engine.

Combines unsupervised Gaussian Mixture / KMeans clustering with an ensemble
Random Forest classifier to detect market regimes, compute state transition
probabilities, and quantify regime stability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


REGIME_NAMES: Dict[int, str] = {
    0: "Bull Trending",
    1: "Bear Trending",
    2: "High Volatility Mean-Reverting",
    3: "Low Volatility Consolidation",
}

REGIME_COLORS: Dict[int, str] = {
    0: "#00ff66",  # Emerald Green
    1: "#ff3344",  # Crimson Red
    2: "#ffaa00",  # Amber Warning
    3: "#00b4d8",  # Ice Blue
}

FEATURE_COLUMNS: List[str] = [
    "Log_Return",
    "Rolling_Vol_20",
    "RSI_14",
    "MACD_Hist",
    "BB_Width",
    "ATR_Pct",
    "Momentum_10",
    "Trend_Ratio_50_200",
]


class MarketRegimeModel:
    """Predictive market regime classification model using ensemble learning."""

    def __init__(self, n_regimes: int = 4, random_state: int = 42) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            random_state=random_state,
            max_iter=200,
        )
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.feature_names = FEATURE_COLUMNS
        self.state_mapping: Dict[int, int] = {}
        self.transition_matrix: np.ndarray = np.zeros((n_regimes, n_regimes))
        self.feature_importances: Dict[str, float] = {}

    def _map_clusters_to_economic_regimes(
        self, features_df: pd.DataFrame, cluster_labels: np.ndarray
    ) -> np.ndarray:
        """Map raw cluster indices to semantic economic regimes.

        Economic definitions:
            - Bull: highest mean return + positive momentum
            - Bear: lowest mean return + negative momentum
            - High Vol: highest rolling volatility among remaining
            - Low Vol: lowest rolling volatility among remaining
        """
        temp = features_df.copy()
        temp["cluster"] = cluster_labels
        stats = temp.groupby("cluster").agg(
            mean_return=("Log_Return", "mean"),
            mean_vol=("Rolling_Vol_20", "mean"),
            mean_mom=("Momentum_10", "mean"),
        )

        clusters = list(range(self.n_regimes))
        # 1. Bull: highest score of return + momentum
        bull_cluster = int(stats["mean_return"].idxmax())

        # 2. Bear: lowest return
        remaining = [c for c in clusters if c != bull_cluster]
        bear_cluster = int(stats.loc[remaining, "mean_return"].idxmin())

        # 3. High Vol from remainder
        remaining = [c for c in remaining if c != bear_cluster]
        if remaining:
            high_vol_cluster = int(stats.loc[remaining, "mean_vol"].idxmax())
            remaining = [c for c in remaining if c != high_vol_cluster]
            low_vol_cluster = int(remaining[0]) if remaining else high_vol_cluster
        else:
            high_vol_cluster = 2
            low_vol_cluster = 3

        self.state_mapping = {
            bull_cluster: 0,
            bear_cluster: 1,
            high_vol_cluster: 2,
            low_vol_cluster: 3,
        }

        mapped = np.array([self.state_mapping.get(c, 0) for c in cluster_labels])
        return mapped

    def fit(self, df: pd.DataFrame) -> "MarketRegimeModel":
        """Fit regime detector on historical feature dataframe.

        Args:
            df: DataFrame containing computed technical features.

        Returns:
            self
        """
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required technical features: {missing_features}")

        x_raw = df[self.feature_names].values
        x_scaled = self.scaler.fit_transform(x_raw)

        # Step 1: Unsupervised latent Gaussian Mixture clustering
        raw_clusters = self.gmm.fit_predict(x_scaled)

        # Step 2: Semantic mapping to economic regimes
        semantic_regimes = self._map_clusters_to_economic_regimes(
            df[self.feature_names], raw_clusters
        )

        # Step 3: Supervised Random Forest calibration on structured features
        self.rf.fit(x_scaled, semantic_regimes)

        # Compute feature importances
        self.feature_importances = {
            name: round(float(imp), 4)
            for name, imp in zip(self.feature_names, self.rf.feature_importances_)
        }

        # Step 4: Empirical Markov transition probability matrix
        self.transition_matrix = self._compute_transition_matrix(semantic_regimes)
        self.is_fitted = True
        return self

    def _compute_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate state transition frequency matrix."""
        matrix = np.zeros((self.n_regimes, self.n_regimes))
        for t in range(len(regimes) - 1):
            i = regimes[t]
            j = regimes[t + 1]
            if 0 <= i < self.n_regimes and 0 <= j < self.n_regimes:
                matrix[i, j] += 1

        # Normalize row-wise to form stochastic transition probabilities
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        normalized = matrix / row_sums
        return np.round(normalized, 4)

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict regime IDs and probability distribution vectors.

        Args:
            df: Technical feature DataFrame.

        Returns:
            Tuple of (predicted_regimes, probabilities).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        x_raw = df[self.feature_names].values
        x_scaled = self.scaler.transform(x_raw)
        preds = self.rf.predict(x_scaled)
        probs = self.rf.predict_proba(x_scaled)
        return preds, probs

    def get_regime_summary(
        self, df: pd.DataFrame, regimes: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Calculate regime statistics and distribution profile."""
        augmented = df.copy()
        augmented["Regime"] = regimes

        summary: List[Dict[str, Any]] = []
        total_samples = len(augmented)

        for reg_id in range(self.n_regimes):
            subset = augmented[augmented["Regime"] == reg_id]
            count = len(subset)
            pct = round((count / max(1, total_samples)) * 100, 2)

            if count > 0:
                mean_daily_ret = float(subset["Log_Return"].mean())
                ann_ret = round(mean_daily_ret * 252 * 100, 2)
                ann_vol = round(float(subset["Log_Return"].std()) * np.sqrt(252) * 100, 2)
                avg_rsi = round(float(subset["RSI_14"].mean()), 2)
                sharpe = round(ann_ret / ann_vol, 2) if ann_vol > 0 else 0.0
            else:
                ann_ret = 0.0
                ann_vol = 0.0
                avg_rsi = 50.0
                sharpe = 0.0

            summary.append(
                {
                    "regime_id": reg_id,
                    "name": REGIME_NAMES[reg_id],
                    "color": REGIME_COLORS[reg_id],
                    "count": count,
                    "percentage": pct,
                    "annualized_return_pct": ann_ret,
                    "annualized_volatility_pct": ann_vol,
                    "sharpe_ratio": sharpe,
                    "avg_rsi": avg_rsi,
                }
            )

        return summary
