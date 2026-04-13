"""Tests for geometry utilities."""

import torch
import pytest

from brain_idp_flow.geometry.se3 import (
    center,
    kabsch_align,
    random_rotation,
    apply_random_rotation,
    rmsd,
)
from brain_idp_flow.geometry.metrics import (
    radius_of_gyration,
    pairwise_distances,
    contact_map,
    end_to_end_distance,
)


def _random_coords(B: int = 2, L: int = 20) -> torch.Tensor:
    coords = torch.randn(B, L, 3)
    return center(coords)


class TestCenter:
    def test_centered_mean_is_zero(self):
        coords = torch.randn(3, 15, 3)
        c = center(coords)
        assert torch.allclose(c.mean(dim=-2), torch.zeros(3, 3), atol=1e-6)


class TestKabsch:
    def test_align_identity(self):
        coords = _random_coords()
        aligned = kabsch_align(coords, coords)
        assert torch.allclose(center(aligned), center(coords), atol=1e-4)

    def test_align_after_rotation(self):
        target = _random_coords(1, 30)
        R = random_rotation(1, target.device)
        mobile = torch.einsum("bij,blj->bli", R, target)

        aligned = kabsch_align(mobile, target)
        r = rmsd(aligned, target)
        assert r.item() < 1e-3

    def test_rmsd_identical_is_zero(self):
        coords = _random_coords()
        r = rmsd(coords, coords)
        assert torch.allclose(r, torch.zeros(2), atol=1e-6)


class TestRandomRotation:
    def test_shape(self):
        R = random_rotation(5, torch.device("cpu"))
        assert R.shape == (5, 3, 3)

    def test_orthogonal(self):
        R = random_rotation(4, torch.device("cpu"))
        I = torch.eye(3).expand(4, -1, -1)
        assert torch.allclose(R @ R.transpose(-1, -2), I, atol=1e-5)

    def test_det_positive(self):
        R = random_rotation(10, torch.device("cpu"))
        dets = torch.det(R)
        assert torch.allclose(dets, torch.ones(10), atol=1e-4)


class TestMetrics:
    def test_rg_shape(self):
        coords = _random_coords(3, 25)
        rg = radius_of_gyration(coords)
        assert rg.shape == (3,)
        assert (rg > 0).all()

    def test_pairwise_distances_symmetric(self):
        coords = _random_coords(2, 10)
        dists = pairwise_distances(coords)
        assert dists.shape == (2, 10, 10)
        assert torch.allclose(dists, dists.transpose(-1, -2), atol=1e-5)

    def test_contact_map_range(self):
        coords = _random_coords(2, 15)
        cm = contact_map(coords, threshold=10.0)
        assert (cm >= 0).all() and (cm <= 1).all()

    def test_e2e_positive(self):
        coords = _random_coords(4, 20)
        e2e = end_to_end_distance(coords)
        assert e2e.shape == (4,)
        assert (e2e >= 0).all()
