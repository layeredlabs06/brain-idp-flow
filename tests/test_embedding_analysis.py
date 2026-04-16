"""Tests for embedding_analysis module (CKA, probes)."""

import numpy as np
import pytest

from brain_idp_flow.analysis.embedding_analysis import (
    linear_cka,
    compute_cross_scale_cka,
    per_layer_rg_probe,
    _center_gram,
    _hsic,
)


class TestLinearCKA:
    """Verify CKA properties per Kornblith et al. 2019."""

    def test_self_similarity_is_one(self):
        X = np.random.RandomState(42).randn(50, 100)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)

    def test_symmetry(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 100)
        Y = rng.randn(50, 80)
        assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-10)

    def test_range_zero_to_one(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 100)
        Y = rng.randn(50, 80)
        cka = linear_cka(X, Y)
        assert 0.0 <= cka <= 1.0

    def test_dissimilar_gram_structures_low_cka(self):
        """Representations with different sample similarity structures
        should yield low CKA."""
        rng = np.random.RandomState(0)
        n = 100
        # X: first half similar to each other, second half random
        X = np.vstack([rng.randn(n // 2, 50) + 5, rng.randn(n // 2, 50) - 5])
        # Y: interleaved structure (odd vs even)
        Y = rng.randn(n, 50)
        Y[::2] += 5
        Y[1::2] -= 5
        cka = linear_cka(X, Y)
        assert cka < 0.5

    def test_different_sample_count_raises(self):
        X = np.random.randn(50, 100)
        Y = np.random.randn(30, 100)
        with pytest.raises(AssertionError):
            linear_cka(X, Y)

    def test_zero_matrix_returns_zero(self):
        X = np.zeros((50, 100))
        Y = np.random.randn(50, 80)
        assert linear_cka(X, Y) == 0.0


class TestCrossScaleCKA:
    def test_output_shape(self):
        rng = np.random.RandomState(42)
        small = {i: rng.randn(20, 480) for i in range(13)}
        large = {i: rng.randn(20, 1280) for i in range(34)}
        matrix = compute_cross_scale_cka(small, large)
        assert matrix.shape == (13, 34)

    def test_diagonal_identity(self):
        """Same representations at matching layers should give CKA=1."""
        rng = np.random.RandomState(42)
        shared = {i: rng.randn(30, 100) for i in range(5)}
        matrix = compute_cross_scale_cka(shared, shared)
        for i in range(5):
            assert matrix[i, i] == pytest.approx(1.0, abs=1e-6)

    def test_values_in_range(self):
        rng = np.random.RandomState(42)
        small = {0: rng.randn(20, 100), 1: rng.randn(20, 100)}
        large = {0: rng.randn(20, 200), 1: rng.randn(20, 200)}
        matrix = compute_cross_scale_cka(small, large)
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0 + 1e-6)


class TestPerLayerRgProbe:
    def test_output_length_matches_layer_count(self):
        rng = np.random.RandomState(42)
        layers = {i: rng.randn(50, 64) for i in range(5)}
        delta_rg = rng.randn(50)
        nuc_scores = rng.randn(50)
        results = per_layer_rg_probe(layers, delta_rg, nuc_scores)
        assert len(results) == 5

    def test_layer_indices_are_sorted(self):
        rng = np.random.RandomState(42)
        layers = {3: rng.randn(50, 64), 0: rng.randn(50, 64), 7: rng.randn(50, 64)}
        delta_rg = rng.randn(50)
        nuc_scores = rng.randn(50)
        results = per_layer_rg_probe(layers, delta_rg, nuc_scores)
        assert [r["layer"] for r in results] == [0, 3, 7]

    def test_rho_values_in_valid_range(self):
        rng = np.random.RandomState(42)
        layers = {0: rng.randn(50, 64), 1: rng.randn(50, 64)}
        delta_rg = rng.randn(50)
        nuc_scores = rng.randn(50)
        results = per_layer_rg_probe(layers, delta_rg, nuc_scores)
        for r in results:
            assert -1.0 <= r["rho_pred_vs_nuc"] <= 1.0
            assert -1.0 <= r["rho_pred_vs_drg"] <= 1.0

    def test_informative_features_produce_nonzero_rho(self):
        """When layer representations directly encode delta_rg,
        the probe should recover a positive correlation."""
        rng = np.random.RandomState(42)
        n = 100
        delta_rg = rng.randn(n)
        nuc_scores = delta_rg + rng.randn(n) * 0.3  # correlated
        # Layer 1: embeds delta_rg directly (with noise)
        X_informative = np.column_stack([delta_rg, rng.randn(n, 9)])
        layers = {0: rng.randn(n, 10), 1: X_informative}
        results = per_layer_rg_probe(layers, delta_rg, nuc_scores)
        layer_1_result = [r for r in results if r["layer"] == 1][0]
        assert layer_1_result["rho_pred_vs_nuc"] > 0.3


class TestCenterGram:
    def test_centered_rows_sum_to_zero(self):
        K = np.random.randn(10, 10)
        K = K @ K.T  # make symmetric PSD
        Kc = _center_gram(K)
        assert np.allclose(Kc.sum(axis=0), 0, atol=1e-10)
        assert np.allclose(Kc.sum(axis=1), 0, atol=1e-10)
