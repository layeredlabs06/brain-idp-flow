"""Hybrid aggregation predictor: ESM-2 embeddings + flow ensemble features.

Combines:
  1. ESM-2 per-residue embeddings (PCA-reduced) — rich sequence info
  2. Flow model structural features — ensemble geometry
  3. ESM-2 LLR — evolutionary constraint

into a single GradientBoosting regressor for aggregation rate prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureContribution:
    """Per-feature contribution to a prediction."""

    name: str
    value: float
    contribution: float


@dataclass(frozen=True)
class HybridPrediction:
    """Result of a single hybrid prediction."""

    score: float
    feature_contributions: tuple[FeatureContribution, ...]
    rg_wt: Optional[np.ndarray] = None
    rg_mut: Optional[np.ndarray] = None


class EnsembleFeatureExtractor:
    """Extract structural features from a flow model ensemble.

    Computes geometry-based features from (K, L, 3) Cα coordinate arrays.
    """

    FEATURE_NAMES = (
        "mean_rg",
        "delta_rg",
        "rg_variance",
        "beta_propensity",
        "contact_entropy",
        "mean_e2e",
        "delta_e2e",
        "site_contact_freq",
    )

    def extract(
        self,
        wt_ensemble: np.ndarray,
        mut_ensemble: np.ndarray,
        mut_pos: int,
    ) -> dict[str, float]:
        """Extract structural features from WT and mutant ensembles.

        Args:
            wt_ensemble: (K, L, 3) WT coordinates
            mut_ensemble: (K, L, 3) mutant coordinates
            mut_pos: 1-based mutation position

        Returns:
            dict of feature name -> value
        """
        from brain_idp_flow.geometry.metrics import (
            radius_of_gyration,
            end_to_end_distance,
            contact_frequency,
            ensemble_rg_variance,
            beta_sheet_propensity,
            contact_entropy,
        )

        wt_t = torch.from_numpy(wt_ensemble).float()
        mut_t = torch.from_numpy(mut_ensemble).float()

        rg_wt = radius_of_gyration(wt_t)
        rg_mut = radius_of_gyration(mut_t)
        e2e_wt = end_to_end_distance(wt_t)
        e2e_mut = end_to_end_distance(mut_t)

        # Site-specific contact frequency change
        cf_wt = contact_frequency(wt_t)
        cf_mut = contact_frequency(mut_t)
        pos_idx = mut_pos - 1  # 0-based
        site_cf_wt = float(cf_wt[pos_idx].mean().item())
        site_cf_mut = float(cf_mut[pos_idx].mean().item())

        return {
            "mean_rg": float(rg_mut.mean().item()),
            "delta_rg": float(rg_mut.mean().item() - rg_wt.mean().item()),
            "rg_variance": ensemble_rg_variance(mut_t),
            "beta_propensity": beta_sheet_propensity(mut_t),
            "contact_entropy": contact_entropy(mut_t),
            "mean_e2e": float(e2e_mut.mean().item()),
            "delta_e2e": float(e2e_mut.mean().item() - e2e_wt.mean().item()),
            "site_contact_freq": site_cf_mut - site_cf_wt,
        }


class HybridAggregationPredictor:
    """Hybrid predictor combining embeddings + structure + LLR.

    Training pipeline:
      1. PCA-reduce ESM-2 embeddings (480d -> n_pca components)
      2. Extract structural features from flow ensembles
      3. Concatenate [PCA embeddings, structural features, LLR]
      4. Fit GradientBoostingRegressor with 5-fold CV

    Args:
        n_pca: number of PCA components for embedding reduction
        n_folds: number of cross-validation folds
        random_state: random seed for reproducibility
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
        self._model = None
        self._feature_names: list[str] = []
        self._cv_results: dict = {}

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        embeddings: np.ndarray,
        structural_features: list[dict[str, float]],
        llr_values: np.ndarray,
        targets: np.ndarray,
    ) -> dict:
        """Fit hybrid model with 5-fold CV evaluation.

        Args:
            embeddings: (N, D) mean-pooled ESM-2 embeddings per mutation
            structural_features: list of N dicts from EnsembleFeatureExtractor
            llr_values: (N,) ESM-2 log-likelihood ratios
            targets: (N,) aggregation rates or nucleation scores

        Returns:
            dict with cv_mean_rho, cv_rhos, train_rho, feature_names
        """
        from sklearn.decomposition import PCA
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler

        # Step 1: PCA on embeddings
        n_components = min(self.n_pca, embeddings.shape[0] - 1, embeddings.shape[1])
        self._pca = PCA(n_components=n_components, random_state=self.random_state)
        emb_pca = self._pca.fit_transform(embeddings)

        # Step 2: Build combined feature matrix
        struct_keys = list(EnsembleFeatureExtractor.FEATURE_NAMES)
        struct_matrix = np.array([
            [sf.get(k, 0.0) for k in struct_keys]
            for sf in structural_features
        ])

        llr_col = llr_values.reshape(-1, 1)

        X = np.hstack([emb_pca, struct_matrix, llr_col])

        pca_names = [f"pca_{i}" for i in range(emb_pca.shape[1])]
        self._feature_names = pca_names + struct_keys + ["llr_site"]

        y = targets.copy()

        # Step 3: 5-fold CV
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

        # Step 4: Fit final model on all data
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

        y_pred_train = self._model.predict(X_scaled)
        train_rho, _ = spearmanr(y_pred_train, y)

        self._cv_results = {
            "model": "hybrid_gb",
            "cv_mean_rho": float(np.mean(cv_rhos)) if cv_rhos else 0.0,
            "cv_rhos": cv_rhos,
            "train_rho": float(train_rho),
            "n": len(y),
            "n_features": X.shape[1],
            "feature_names": self._feature_names,
            "pca_variance_explained": float(self._pca.explained_variance_ratio_.sum()),
        }

        self._print_results()
        return self._cv_results

    def _print_results(self) -> None:
        r = self._cv_results
        print(f"\n{'='*60}")
        print(f"HYBRID AGGREGATION PREDICTOR (n={r['n']}, p={r['n_features']})")
        print(f"{'='*60}")
        print(f"PCA variance explained: {r['pca_variance_explained']:.1%}")
        print(f"Train ρ: {r['train_rho']:.3f}")
        print(f"CV ρ (mean {self.n_folds}-fold): {r['cv_mean_rho']:.3f}")
        if r["cv_rhos"]:
            print(f"CV ρ per fold: {[f'{x:.3f}' for x in r['cv_rhos']]}")

        # Top feature importances
        if self._model is not None:
            importances = self._model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            print(f"\nTop features:")
            for i in top_idx:
                name = self._feature_names[i] if i < len(self._feature_names) else f"f{i}"
                print(f"  {name:<30} {importances[i]:.4f}")

    def predict_single(
        self,
        embedding: np.ndarray,
        structural_features: dict[str, float],
        llr_value: float,
    ) -> float:
        """Predict aggregation score for a single mutation.

        Args:
            embedding: (D,) mean-pooled ESM-2 embedding
            structural_features: dict from EnsembleFeatureExtractor
            llr_value: ESM-2 LLR for this mutation

        Returns:
            predicted aggregation score
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        emb_pca = self._pca.transform(embedding.reshape(1, -1))

        struct_keys = list(EnsembleFeatureExtractor.FEATURE_NAMES)
        struct_vec = np.array([[structural_features.get(k, 0.0) for k in struct_keys]])

        X = np.hstack([emb_pca, struct_vec, [[llr_value]]])
        X_scaled = self._scaler.transform(X)

        return float(self._model.predict(X_scaled)[0])

    def explain(
        self,
        embedding: np.ndarray,
        structural_features: dict[str, float],
        llr_value: float,
    ) -> list[FeatureContribution]:
        """Estimate feature contributions for a single prediction.

        Uses feature importances * feature deviation from training mean
        as a simple attribution method.

        Returns:
            sorted list of FeatureContribution (highest abs contribution first)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        emb_pca = self._pca.transform(embedding.reshape(1, -1))
        struct_keys = list(EnsembleFeatureExtractor.FEATURE_NAMES)
        struct_vec = np.array([[structural_features.get(k, 0.0) for k in struct_keys]])
        X = np.hstack([emb_pca, struct_vec, [[llr_value]]])
        X_scaled = self._scaler.transform(X)

        importances = self._model.feature_importances_
        contributions = []

        for i, name in enumerate(self._feature_names):
            contrib = float(importances[i] * abs(X_scaled[0, i]))
            contributions.append(FeatureContribution(
                name=name,
                value=float(X[0, i]),
                contribution=contrib,
            ))

        return sorted(contributions, key=lambda c: abs(c.contribution), reverse=True)

    def evaluate_ood(
        self,
        embeddings: np.ndarray,
        structural_features: list[dict[str, float]],
        llr_values: np.ndarray,
        targets: np.ndarray,
        dataset_name: str = "OOD",
    ) -> dict:
        """Evaluate on out-of-distribution dataset (e.g., IAPP).

        Returns:
            dict with rho, p_value, n, predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_components = self._pca.n_components_
        emb_pca = self._pca.transform(embeddings)

        struct_keys = list(EnsembleFeatureExtractor.FEATURE_NAMES)
        struct_matrix = np.array([
            [sf.get(k, 0.0) for k in struct_keys]
            for sf in structural_features
        ])

        X = np.hstack([emb_pca, struct_matrix, llr_values.reshape(-1, 1)])
        X_scaled = self._scaler.transform(X)

        predictions = self._model.predict(X_scaled)
        rho, pval = spearmanr(predictions, targets)

        print(f"\n{dataset_name} evaluation: ρ={rho:.3f}, p={pval:.4f} (n={len(targets)})")

        return {
            "dataset": dataset_name,
            "rho": float(rho),
            "p_value": float(pval),
            "n": len(targets),
            "predictions": predictions,
        }
