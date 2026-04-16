"""ESM-2 layer-wise embedding analysis: CKA, per-layer probes.

Answers the key question: *why* does the delta-Rg/nucleation-score
correlation reverse between small (35M) and large (650M) PLMs?

Hypothesis: early layers are scale-invariant (local structure, positive
correlation with expansion), while late layers at 650M encode long-range
contacts that invert the compaction signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment) — Kornblith et al., ICML 2019
# ---------------------------------------------------------------------------

def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix: H K H, where H = I - 1/n."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion (biased estimator)."""
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    return float(np.sum(Kc * Lc))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    Uses kernel CKA with linear kernels so that X and Y can have
    different feature dimensions (d1 != d2).  This is the biased
    HSIC estimator; the ratio HSIC(K,L)/sqrt(HSIC(K,K)*HSIC(L,L))
    is equivalent to Kornblith et al. 2019 Eq. 5 (the constant
    cancels in the normalization).

    Args:
        X: (n, d1) representation matrix
        Y: (n, d2) representation matrix

    Returns:
        CKA similarity in [0, 1].
    """
    assert X.shape[0] == Y.shape[0], "Same number of samples required"
    K = X @ X.T
    L = Y @ Y.T
    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-12:
        return 0.0
    return float(hsic_kl / denom)


# ---------------------------------------------------------------------------
# Cross-scale CKA matrix
# ---------------------------------------------------------------------------

def compute_cross_scale_cka(
    layers_small: dict[int, np.ndarray],
    layers_large: dict[int, np.ndarray],
) -> np.ndarray:
    """Compute CKA between every layer pair across two model scales.

    Args:
        layers_small: {layer_idx: (n_seqs, d_small)} mean-pooled representations
        layers_large: {layer_idx: (n_seqs, d_large)} mean-pooled representations

    Returns:
        (n_layers_small, n_layers_large) CKA matrix.
    """
    small_keys = sorted(layers_small.keys())
    large_keys = sorted(layers_large.keys())

    cka_matrix = np.zeros((len(small_keys), len(large_keys)))
    for i, sk in enumerate(small_keys):
        for j, lk in enumerate(large_keys):
            cka_matrix[i, j] = linear_cka(layers_small[sk], layers_large[lk])

    return cka_matrix


def extract_mean_pooled_layers(
    embedder,
    sequences: list[str],
    batch_size: int = 8,
) -> dict[int, np.ndarray]:
    """Extract mean-pooled per-layer representations for a set of sequences.

    Args:
        embedder: ESM2Embedder instance
        sequences: list of amino acid strings
        batch_size: process this many sequences at a time

    Returns:
        {layer_idx: (n_seqs, d)} numpy arrays, mean-pooled over residues.
    """
    import torch

    all_layers: dict[int, list[np.ndarray]] = {}

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        layer_dict = embedder.embed_all_layers(batch)

        for layer_idx, tensor in layer_dict.items():
            # Mean-pool over residue dimension: (B, L, D) -> (B, D)
            pooled = tensor.mean(dim=1).cpu().numpy()
            if layer_idx not in all_layers:
                all_layers[layer_idx] = []
            all_layers[layer_idx].append(pooled)

    return {
        layer: np.concatenate(chunks, axis=0)
        for layer, chunks in all_layers.items()
    }


# ---------------------------------------------------------------------------
# Per-layer linear probe for delta-Rg correlation
# ---------------------------------------------------------------------------

def per_layer_rg_probe(
    layer_representations: dict[int, np.ndarray],
    delta_rg: np.ndarray,
    nucleation_scores: np.ndarray,
    n_folds: int = 5,
) -> list[dict]:
    """Train a linear probe per ESM-2 layer to predict delta-Rg,
    then correlate predictions with nucleation scores.

    This reveals which layers drive the positive vs negative correlation.

    Args:
        layer_representations: {layer_idx: (n_mutations, d)} from mean-pooled
            embeddings of mutant sequences.
        delta_rg: (n_mutations,) flow-model delta-Rg values.
        nucleation_scores: (n_mutations,) experimental nucleation scores.
        n_folds: cross-validation folds.

    Returns:
        List of dicts with layer_idx, rho_pred_vs_nuc, rho_pred_vs_drg, cv_r2.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    for layer_idx in sorted(layer_representations.keys()):
        X = layer_representations[layer_idx]

        # CV predictions of delta-Rg from embeddings
        # Scaler must be fit inside each fold to avoid data leakage
        pred_drg = np.zeros(len(delta_rg))
        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            model = Ridge(alpha=1.0)
            model.fit(X_tr, delta_rg[train_idx])
            pred_drg[test_idx] = model.predict(X_te)

        rho_vs_nuc, p_vs_nuc = spearmanr(pred_drg, nucleation_scores)
        rho_vs_drg, _ = spearmanr(pred_drg, delta_rg)

        results.append({
            "layer": layer_idx,
            "rho_pred_vs_nuc": float(rho_vs_nuc),
            "p_pred_vs_nuc": float(p_vs_nuc),
            "rho_pred_vs_drg": float(rho_vs_drg),
        })

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    small_label: str = "ESM-2 35M",
    large_label: str = "ESM-2 650M",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot CKA similarity heatmap between two model scales."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cka_matrix.T, origin="lower", aspect="auto",
                   cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel(f"{small_label} layer")
    ax.set_ylabel(f"{large_label} layer")
    ax.set_title(f"Cross-Scale CKA: {small_label} vs {large_label}")
    fig.colorbar(im, ax=ax, label="Linear CKA")

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_layer_rho_curve(
    probe_results_small: list[dict],
    probe_results_large: list[dict],
    small_label: str = "ESM-2 35M",
    large_label: str = "ESM-2 650M",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot per-layer Spearman rho with nucleation score for two scales.

    The crossover point where 650M goes negative is the key finding.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    # Normalize layer indices to [0, 1] for comparison across scales
    for results, label, color in [
        (probe_results_small, small_label, "#3498db"),
        (probe_results_large, large_label, "#e74c3c"),
    ]:
        layers = [r["layer"] for r in results]
        rhos = [r["rho_pred_vs_nuc"] for r in results]
        max_layer = max(layers) if layers else 1
        normalized = [l / max_layer for l in layers]

        ax.plot(normalized, rhos, "o-", color=color, label=label,
                markersize=4, linewidth=1.5)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Normalized layer depth (0=input, 1=final)")
    ax.set_ylabel("Spearman rho (predicted delta-Rg vs nucleation score)")
    ax.set_title("Per-Layer Correlation Direction: Scale-Dependent Reversal")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)
