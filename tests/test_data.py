"""Tests for data loading utilities."""

import numpy as np
import torch
import pytest
from pathlib import Path

from brain_idp_flow.targets import load_targets, Target, Mutation
from brain_idp_flow.geometry.metrics import js_divergence_1d


class TestTargets:
    def test_load_targets(self, tmp_path):
        config = tmp_path / "targets.yaml"
        config.write_text("""
targets:
  test_protein:
    name: "Test Protein"
    uniprot: "P12345"
    region: [1, 10]
    length: 10
    ped_id: "PED00001"
    disease: "Test"
    sequence: "ACDEFGHIKL"
    mutations:
      - { id: "A1G", pos: 1, wt: "A", mt: "G", agg_rate_relative: 2.0 }
""")
        targets = load_targets(config)
        assert "test_protein" in targets
        t = targets["test_protein"]
        assert t.length == 10
        assert len(t.mutations) == 1
        assert t.mutations[0].id == "A1G"

    def test_mutant_sequence(self):
        t = Target(
            id="test", name="Test", uniprot="P0",
            region=(1, 5), length=5, ped_id="PED0",
            disease="Test", sequence="ACDEF",
            mutations=(Mutation("A1G", 1, "A", "G"),),
        )
        mut_seq = t.mutant_sequence(t.mutations[0])
        assert mut_seq == "GCDEF"

    def test_mutant_sequence_wrong_wt_raises(self):
        t = Target(
            id="test", name="Test", uniprot="P0",
            region=(1, 5), length=5, ped_id="PED0",
            disease="Test", sequence="ACDEF",
            mutations=(Mutation("X1G", 1, "X", "G"),),  # wrong WT
        )
        with pytest.raises(AssertionError):
            t.mutant_sequence(t.mutations[0])


class TestJSDivergence:
    def test_identical_distributions(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 1000)
        js = js_divergence_1d(samples, samples)
        assert js < 0.05  # should be near zero

    def test_different_distributions(self):
        rng = np.random.default_rng(42)
        p = rng.normal(0, 1, 1000)
        q = rng.normal(5, 1, 1000)
        js = js_divergence_1d(p, q)
        assert js > 0.3  # should be large


class TestDataset:
    def test_dataset_from_npz(self, tmp_path):
        from brain_idp_flow.data.dataset import ProteinEnsembleDataset

        n, L = 20, 15
        coords = np.random.randn(n, L, 3).astype(np.float32)
        seq_ids = np.zeros(n, dtype=np.int64)
        mut_pos = np.zeros(n, dtype=np.int64)
        mut_aa = np.zeros(n, dtype=np.int64)

        npz_path = tmp_path / "test.npz"
        np.savez(npz_path, coords=coords, seq_ids=seq_ids,
                 mut_pos=mut_pos, mut_aa=mut_aa)

        ds = ProteinEnsembleDataset(npz_path, max_len=20)
        assert len(ds) == n

        item = ds[0]
        assert item["coords"].shape == (L, 3)
        assert item["seq_id"].item() == 0
