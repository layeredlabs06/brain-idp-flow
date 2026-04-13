"""Structural metrics for protein ensembles."""

from __future__ import annotations

import torch
from torch import Tensor
import numpy as np


def radius_of_gyration(coords: Tensor) -> Tensor:
    """Radius of gyration for Cα coordinates.

    coords: (..., L, 3)  ->  (...,)
    """
    centered = coords - coords.mean(dim=-2, keepdim=True)
    return torch.sqrt((centered ** 2).sum(dim=-1).mean(dim=-1))


def pairwise_distances(coords: Tensor) -> Tensor:
    """Pairwise Cα distance matrix.

    coords: (..., L, 3) -> (..., L, L)
    """
    diff = coords.unsqueeze(-2) - coords.unsqueeze(-3)
    return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)


def contact_map(coords: Tensor, threshold: float = 8.0) -> Tensor:
    """Binary contact map (Cα distance < threshold).

    coords: (..., L, 3) -> (..., L, L) float [0, 1]
    """
    dists = pairwise_distances(coords)
    return (dists < threshold).float()


def contact_frequency(ensemble: Tensor, threshold: float = 8.0) -> Tensor:
    """Mean contact frequency across an ensemble.

    ensemble: (K, L, 3) -> (L, L)
    """
    contacts = contact_map(ensemble, threshold)
    return contacts.mean(dim=0)


def end_to_end_distance(coords: Tensor) -> Tensor:
    """Distance between first and last Cα.

    coords: (..., L, 3) -> (...,)
    """
    diff = coords[..., -1, :] - coords[..., 0, :]
    return torch.sqrt((diff ** 2).sum(dim=-1))


def js_divergence_1d(p_samples: np.ndarray, q_samples: np.ndarray,
                     n_bins: int = 50) -> float:
    """Jensen-Shannon divergence between two 1D sample distributions."""
    all_samples = np.concatenate([p_samples, q_samples])
    bins = np.linspace(all_samples.min(), all_samples.max(), n_bins + 1)

    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)

    # Normalize to probability
    p_hist = p_hist / (p_hist.sum() + 1e-10)
    q_hist = q_hist / (q_hist.sum() + 1e-10)

    m = 0.5 * (p_hist + q_hist)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p_hist, m) + 0.5 * _kl(q_hist, m)
