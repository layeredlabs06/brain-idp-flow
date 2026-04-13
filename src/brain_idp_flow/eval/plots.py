"""Visualization utilities for protein ensembles."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import torch

from brain_idp_flow.geometry.metrics import (
    radius_of_gyration,
    contact_frequency,
)


def plot_rg_comparison(
    ensembles: dict[str, np.ndarray],
    title: str = "Radius of Gyration",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot Rg histograms for multiple ensembles."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, coords in ensembles.items():
        rg = radius_of_gyration(torch.from_numpy(coords)).numpy()
        ax.hist(rg, bins=30, alpha=0.5, label=f"{label} (μ={rg.mean():.1f})", density=True)

    ax.set_xlabel("Radius of Gyration (Å)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
    return fig


def plot_contact_maps(
    ensembles: dict[str, np.ndarray],
    title: str = "Contact Frequency Maps",
    threshold: float = 8.0,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot contact frequency maps side by side."""
    n = len(ensembles)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (label, coords) in zip(axes, ensembles.items()):
        cf = contact_frequency(torch.from_numpy(coords), threshold).numpy()
        im = ax.imshow(cf, cmap="hot", vmin=0, vmax=1)
        ax.set_title(label)
        ax.set_xlabel("Residue")
        ax.set_ylabel("Residue")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
    return fig


def plot_3d_traces(
    ensemble: np.ndarray,
    n_traces: int = 10,
    title: str = "Cα Backbone Traces",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot 3D Cα backbone traces for an ensemble."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    indices = np.random.choice(len(ensemble), min(n_traces, len(ensemble)), replace=False)

    for i in indices:
        coords = ensemble[i]
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], alpha=0.4, linewidth=0.8)

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title)

    if save_path:
        fig.savefig(str(save_path), dpi=150)
    return fig


def plot_mutation_comparison(
    wt_ensemble: np.ndarray,
    mut_ensembles: dict[str, np.ndarray],
    reference: Optional[np.ndarray] = None,
    target_name: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Compare WT vs multiple mutants: Rg distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # WT
    rg_wt = radius_of_gyration(torch.from_numpy(wt_ensemble)).numpy()
    ax.hist(rg_wt, bins=30, alpha=0.4, label=f"WT (μ={rg_wt.mean():.1f})", density=True)

    # Reference
    if reference is not None:
        min_len = min(reference.shape[1], wt_ensemble.shape[1])
        rg_ref = radius_of_gyration(torch.from_numpy(reference[:, :min_len])).numpy()
        ax.hist(rg_ref, bins=30, alpha=0.3, label=f"PED ref (μ={rg_ref.mean():.1f})",
                density=True, histtype="step", linewidth=2)

    # Mutants
    colors = plt.cm.Set1(np.linspace(0, 1, len(mut_ensembles)))
    for (mut_id, coords), color in zip(mut_ensembles.items(), colors):
        rg = radius_of_gyration(torch.from_numpy(coords)).numpy()
        ax.hist(rg, bins=30, alpha=0.3, label=f"{mut_id} (μ={rg.mean():.1f})",
                density=True, color=color)

    ax.set_xlabel("Radius of Gyration (Å)")
    ax.set_ylabel("Density")
    ax.set_title(f"{target_name} — WT vs Mutant Rg Distributions")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
    return fig
