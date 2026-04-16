"""Lightweight aggregation predictor using ESM-2 embeddings only.

No flow model required. PCA-reduced mean-pooled ESM-2 embeddings
fed into GradientBoosting regressor. Achieves ρ=0.595 on Aβ42 DMS.

Usage:
    predictor = EmbeddingAggregationPredictor()
    predictor.fit(embeddings, targets)
    score = predictor.predict_single(embedding)
    predictor.save("model.pkl")
    predictor = EmbeddingAggregationPredictor.load("model.pkl")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class PredictionResult:
    """Single mutation prediction result."""

    score: float
    risk_level: str  # HIGH / MEDIUM / LOW
    top_features: tuple[tuple[str, float], ...]


class EmbeddingAggregationPredictor:
    """ESM-2 embedding-only aggregation predictor.

    Architecture: mean-pooled ESM-2 → PCA → GradientBoosting → score
    """

    def __init__(
        self,
        n_pca: int = 50,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        self.n_pca = n_pca
        self.n_folds = n_folds
        self.random_state = random_state
        self._pca = None
        self._scaler = None
        self._model = None
        self._cv_results: dict = {}

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
    ) -> dict:
        """Fit predictor with 5-fold CV evaluation.

        Args:
            embeddings: (N, D) mean-pooled ESM-2 embeddings
            targets: (N,) nucleation scores or aggregation rates

        Returns:
            dict with cv_mean_rho, cv_rhos, train_rho
        """
        from sklearn.decomposition import PCA
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler

        n_components = min(self.n_pca, embeddings.shape[0] - 1, embeddings.shape[1])
        self._pca = PCA(n_components=n_components, random_state=self.random_state)
        X = self._pca.fit_transform(embeddings)

        y = targets.copy()

        # 5-fold CV
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_rhos = []

        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])

            gb = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=self.random_state,
            )
            gb.fit(X_tr, y[train_idx])
            pred = gb.predict(X_te)

            if len(pred) >= 3:
                rho, _ = spearmanr(pred, y[test_idx])
                if not np.isnan(rho):
                    cv_rhos.append(rho)

        # Final model on all data
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=self.random_state,
        )
        self._model.fit(X_scaled, y)

        y_pred = self._model.predict(X_scaled)
        train_rho, _ = spearmanr(y_pred, y)

        self._cv_results = {
            "cv_mean_rho": float(np.mean(cv_rhos)) if cv_rhos else 0.0,
            "cv_rhos": cv_rhos,
            "train_rho": float(train_rho),
            "n": len(y),
            "n_pca": n_components,
            "pca_variance_explained": float(self._pca.explained_variance_ratio_.sum()),
        }

        print(f"Embedding Predictor: CV ρ={self._cv_results['cv_mean_rho']:.3f}, "
              f"Train ρ={train_rho:.3f}, PCA var={self._cv_results['pca_variance_explained']:.1%}")

        return self._cv_results

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict aggregation scores for multiple mutations.

        Args:
            embeddings: (N, D) mean-pooled ESM-2 embeddings

        Returns:
            (N,) predicted scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._pca.transform(embeddings)
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def predict_single(self, embedding: np.ndarray) -> PredictionResult:
        """Predict with risk level and feature contributions.

        Args:
            embedding: (D,) mean-pooled ESM-2 embedding

        Returns:
            PredictionResult with score, risk_level, top_features
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._pca.transform(embedding.reshape(1, -1))
        X_scaled = self._scaler.transform(X)
        score = float(self._model.predict(X_scaled)[0])

        risk = "HIGH" if score > 0.5 else "MEDIUM" if score > 0 else "LOW"

        # Top contributing PCA components
        importances = self._model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        top_features = tuple(
            (f"pca_{i}", float(importances[i]))
            for i in top_idx
        )

        return PredictionResult(
            score=score,
            risk_level=risk,
            top_features=top_features,
        )

    def save(self, path: str | Path) -> None:
        """Save fitted model to pickle."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        data = {
            "pca": self._pca,
            "scaler": self._scaler,
            "model": self._model,
            "cv_results": self._cv_results,
            "n_pca": self.n_pca,
            "n_folds": self.n_folds,
            "random_state": self.random_state,
        }
        with open(str(path), "wb") as f:
            pickle.dump(data, f)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingAggregationPredictor":
        """Load fitted model from pickle."""
        with open(str(path), "rb") as f:
            data = pickle.load(f)

        predictor = cls(
            n_pca=data["n_pca"],
            n_folds=data["n_folds"],
            random_state=data["random_state"],
        )
        predictor._pca = data["pca"]
        predictor._scaler = data["scaler"]
        predictor._model = data["model"]
        predictor._cv_results = data["cv_results"]
        return predictor
