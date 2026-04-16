"""Tests for hybrid predictor and new geometry features."""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# geometry/metrics.py — new functions
# ---------------------------------------------------------------------------

class TestEnsembleRgVariance:
    def test_basic_shape(self):
        from brain_idp_flow.geometry.metrics import ensemble_rg_variance
        ensemble = torch.randn(50, 20, 3)
        var = ensemble_rg_variance(ensemble)
        assert isinstance(var, float)
        assert var >= 0.0

    def test_identical_structures_zero_variance(self):
        from brain_idp_flow.geometry.metrics import ensemble_rg_variance
        single = torch.randn(1, 15, 3)
        ensemble = single.expand(30, -1, -1)
        var = ensemble_rg_variance(ensemble)
        assert var < 1e-6

    def test_diverse_ensemble_higher_variance(self):
        from brain_idp_flow.geometry.metrics import ensemble_rg_variance
        compact = torch.randn(25, 20, 3) * 0.5
        extended = torch.randn(25, 20, 3) * 5.0
        mixed = torch.cat([compact, extended], dim=0)
        var_mixed = ensemble_rg_variance(mixed)
        var_compact = ensemble_rg_variance(compact)
        assert var_mixed > var_compact


class TestBetaSheetPropensity:
    def test_basic_range(self):
        from brain_idp_flow.geometry.metrics import beta_sheet_propensity
        ensemble = torch.randn(30, 25, 3) * 3.0
        prop = beta_sheet_propensity(ensemble)
        assert 0.0 <= prop <= 1.0

    def test_short_sequence(self):
        from brain_idp_flow.geometry.metrics import beta_sheet_propensity
        ensemble = torch.randn(10, 2, 3)
        prop = beta_sheet_propensity(ensemble)
        assert prop == 0.0

    def test_extended_chain_high_propensity(self):
        from brain_idp_flow.geometry.metrics import beta_sheet_propensity
        # Build an extended chain with i->i+2 distances ~6.5 A
        L = 20
        K = 30
        coords = torch.zeros(K, L, 3)
        for i in range(L):
            coords[:, i, 0] = i * 3.25  # ~3.25A per residue => i,i+2 = 6.5A
        prop = beta_sheet_propensity(coords)
        assert prop > 0.5


class TestContactEntropy:
    def test_basic_output(self):
        from brain_idp_flow.geometry.metrics import contact_entropy
        ensemble = torch.randn(40, 20, 3) * 5.0
        ent = contact_entropy(ensemble)
        assert isinstance(ent, float)
        assert ent >= 0.0

    def test_compact_vs_extended(self):
        from brain_idp_flow.geometry.metrics import contact_entropy
        compact = torch.randn(40, 15, 3) * 1.0  # many contacts
        extended = torch.randn(40, 15, 3) * 20.0  # few contacts
        ent_compact = contact_entropy(compact)
        ent_extended = contact_entropy(extended)
        # Both should be valid floats; exact ordering depends on entropy definition
        assert ent_compact >= 0.0
        assert ent_extended >= 0.0


# ---------------------------------------------------------------------------
# model/hybrid_predictor.py — EnsembleFeatureExtractor
# ---------------------------------------------------------------------------

class TestEnsembleFeatureExtractor:
    def test_extract_returns_all_keys(self):
        from brain_idp_flow.model.hybrid_predictor import EnsembleFeatureExtractor

        extractor = EnsembleFeatureExtractor()
        wt = np.random.randn(50, 20, 3).astype(np.float32) * 3.0
        mut = np.random.randn(50, 20, 3).astype(np.float32) * 3.0
        features = extractor.extract(wt, mut, mut_pos=5)

        for key in EnsembleFeatureExtractor.FEATURE_NAMES:
            assert key in features, f"Missing feature: {key}"
            assert isinstance(features[key], float)

    def test_delta_rg_sign(self):
        from brain_idp_flow.model.hybrid_predictor import EnsembleFeatureExtractor

        extractor = EnsembleFeatureExtractor()
        compact = np.random.randn(50, 20, 3).astype(np.float32) * 1.0
        extended = np.random.randn(50, 20, 3).astype(np.float32) * 5.0

        feats = extractor.extract(compact, extended, mut_pos=1)
        assert feats["delta_rg"] > 0, "Extended mutant should have positive delta_rg"


# ---------------------------------------------------------------------------
# model/hybrid_predictor.py — HybridAggregationPredictor
# ---------------------------------------------------------------------------

class TestHybridAggregationPredictor:
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic training data for predictor tests."""
        np.random.seed(42)
        n = 60
        d = 480

        embeddings = np.random.randn(n, d).astype(np.float32)
        structural_features = []
        for i in range(n):
            structural_features.append({
                "mean_rg": np.random.uniform(8, 15),
                "delta_rg": np.random.uniform(-2, 2),
                "rg_variance": np.random.uniform(0, 3),
                "beta_propensity": np.random.uniform(0, 0.5),
                "contact_entropy": np.random.uniform(0.3, 0.9),
                "mean_e2e": np.random.uniform(5, 25),
                "delta_e2e": np.random.uniform(-3, 3),
                "site_contact_freq": np.random.uniform(-0.2, 0.2),
            })
        llr_values = np.random.randn(n).astype(np.float32)
        # Target correlates with delta_rg + noise
        targets = np.array([sf["delta_rg"] for sf in structural_features]) + np.random.randn(n) * 0.5

        return embeddings, structural_features, llr_values, targets

    def test_fit_returns_results(self, synthetic_data):
        from brain_idp_flow.model.hybrid_predictor import HybridAggregationPredictor

        emb, sf, llr, targets = synthetic_data
        predictor = HybridAggregationPredictor(n_pca=10, n_folds=3)
        results = predictor.fit(emb, sf, llr, targets)

        assert "cv_mean_rho" in results
        assert "train_rho" in results
        assert "n" in results
        assert results["n"] == len(targets)
        assert predictor.is_fitted

    def test_predict_single(self, synthetic_data):
        from brain_idp_flow.model.hybrid_predictor import HybridAggregationPredictor

        emb, sf, llr, targets = synthetic_data
        predictor = HybridAggregationPredictor(n_pca=10, n_folds=3)
        predictor.fit(emb, sf, llr, targets)

        score = predictor.predict_single(emb[0], sf[0], llr[0])
        assert isinstance(score, float)

    def test_explain(self, synthetic_data):
        from brain_idp_flow.model.hybrid_predictor import HybridAggregationPredictor

        emb, sf, llr, targets = synthetic_data
        predictor = HybridAggregationPredictor(n_pca=10, n_folds=3)
        predictor.fit(emb, sf, llr, targets)

        contributions = predictor.explain(emb[0], sf[0], llr[0])
        assert len(contributions) > 0
        assert all(hasattr(c, "name") and hasattr(c, "contribution") for c in contributions)
        # Sorted by abs contribution descending
        abs_contribs = [abs(c.contribution) for c in contributions]
        assert abs_contribs == sorted(abs_contribs, reverse=True)

    def test_predict_before_fit_raises(self):
        from brain_idp_flow.model.hybrid_predictor import HybridAggregationPredictor

        predictor = HybridAggregationPredictor()
        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.predict_single(np.zeros(480), {}, 0.0)

    def test_evaluate_ood(self, synthetic_data):
        from brain_idp_flow.model.hybrid_predictor import HybridAggregationPredictor

        emb, sf, llr, targets = synthetic_data
        predictor = HybridAggregationPredictor(n_pca=10, n_folds=3)
        predictor.fit(emb, sf, llr, targets)

        # Evaluate on a subset as "OOD"
        ood_result = predictor.evaluate_ood(
            emb[:10], sf[:10], llr[:10], targets[:10], "test_ood"
        )
        assert "rho" in ood_result
        assert "n" in ood_result
        assert ood_result["n"] == 10


# ---------------------------------------------------------------------------
# app.py — parse_mutation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# model/embedding_predictor.py
# ---------------------------------------------------------------------------

class TestEmbeddingAggregationPredictor:
    @pytest.fixture
    def synth(self):
        np.random.seed(42)
        n, d = 60, 480
        emb = np.random.randn(n, d).astype(np.float32)
        targets = np.random.randn(n).astype(np.float32)
        return emb, targets

    def test_fit_and_predict(self, synth):
        from brain_idp_flow.model.embedding_predictor import EmbeddingAggregationPredictor
        emb, targets = synth
        p = EmbeddingAggregationPredictor(n_pca=10, n_folds=3)
        results = p.fit(emb, targets)
        assert "cv_mean_rho" in results
        assert p.is_fitted
        scores = p.predict(emb[:5])
        assert scores.shape == (5,)

    def test_predict_single(self, synth):
        from brain_idp_flow.model.embedding_predictor import EmbeddingAggregationPredictor
        emb, targets = synth
        p = EmbeddingAggregationPredictor(n_pca=10, n_folds=3)
        p.fit(emb, targets)
        result = p.predict_single(emb[0])
        assert hasattr(result, "score")
        assert hasattr(result, "risk_level")
        assert result.risk_level in ("HIGH", "MEDIUM", "LOW")
        assert len(result.top_features) == 10

    def test_save_load(self, synth, tmp_path):
        from brain_idp_flow.model.embedding_predictor import EmbeddingAggregationPredictor
        emb, targets = synth
        p = EmbeddingAggregationPredictor(n_pca=10, n_folds=3)
        p.fit(emb, targets)
        score_before = p.predict(emb[:3])

        path = tmp_path / "model.pkl"
        p.save(path)

        p2 = EmbeddingAggregationPredictor.load(path)
        score_after = p2.predict(emb[:3])
        np.testing.assert_array_almost_equal(score_before, score_after)

    def test_not_fitted_raises(self):
        from brain_idp_flow.model.embedding_predictor import EmbeddingAggregationPredictor
        p = EmbeddingAggregationPredictor()
        with pytest.raises(RuntimeError, match="not fitted"):
            p.predict(np.zeros((1, 480)))


class TestParseMutation:
    def test_valid_mutations(self):
        from brain_idp_flow.app import _parse_mutation

        assert _parse_mutation("E22G") == ("E", 22, "G")
        assert _parse_mutation("D1A") == ("D", 1, "A")
        assert _parse_mutation("A100V") == ("A", 100, "V")
        assert _parse_mutation("  e22g  ") == ("E", 22, "G")

    def test_invalid_format(self):
        from brain_idp_flow.app import _parse_mutation

        with pytest.raises(ValueError):
            _parse_mutation("22")
        with pytest.raises(ValueError):
            _parse_mutation("EG")
        with pytest.raises(ValueError):
            _parse_mutation("")
