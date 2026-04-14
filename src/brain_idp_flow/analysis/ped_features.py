"""Extract structural features from PED ensembles at mutation sites."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from brain_idp_flow.geometry.metrics import (
    pairwise_distances,
    contact_map,
    radius_of_gyration,
)


def rmsf(ensemble: np.ndarray) -> np.ndarray:
    """Per-residue RMSF from ensemble. ensemble: (K, L, 3) -> (L,)."""
    mean_pos = ensemble.mean(axis=0)  # (L, 3)
    deviations = ensemble - mean_pos[None, :, :]  # (K, L, 3)
    return np.sqrt((deviations ** 2).sum(axis=-1).mean(axis=0))


def extract_mutation_site_features(
    ensemble: np.ndarray,
    mutation_pos: int,
    window: int = 5,
) -> dict:
    """Extract structural features at a mutation site from PED ensemble.

    Args:
        ensemble: (K, L, 3) Cα coordinates
        mutation_pos: 0-indexed position in the ensemble
        window: residues on each side for local features

    Returns dict of features.
    """
    K, L, _ = ensemble.shape
    pos = min(mutation_pos, L - 1)  # clamp to valid range

    ens_t = torch.from_numpy(ensemble)

    # Per-residue RMSF (flexibility)
    residue_rmsf = rmsf(ensemble)
    site_rmsf = float(residue_rmsf[pos])

    # Contact frequency at mutation site
    contacts = contact_map(ens_t, threshold=8.0)  # (K, L, L)
    site_contact_freq = float(contacts[:, pos, :].mean())

    # Long-range contacts (|i-j| > 10)
    long_range_mask = torch.abs(torch.arange(L) - pos) > 10
    site_long_range_cf = float(contacts[:, pos, :][:, long_range_mask].mean())

    # Local Rg (window around mutation site)
    start = max(0, pos - window)
    end = min(L, pos + window + 1)
    local_coords = ens_t[:, start:end, :]
    local_rg = float(radius_of_gyration(local_coords).mean())

    # Beta-bridge propensity: i,i+2 contact frequency (proxy)
    beta_proxy = 0.0
    if pos + 2 < L:
        dists = torch.sqrt(((ens_t[:, pos, :] - ens_t[:, pos + 2, :]) ** 2).sum(-1))
        beta_proxy = float((dists < 7.0).float().mean())

    # Solvent exposure proxy: inverse local contact density
    local_contacts = float(contacts[:, pos, max(0, pos-5):min(L, pos+6)].mean())
    exposure = 1.0 / (local_contacts + 0.01)

    return {
        "site_rmsf": site_rmsf,
        "site_contact_freq": site_contact_freq,
        "site_long_range_cf": site_long_range_cf,
        "local_rg": local_rg,
        "beta_propensity": beta_proxy,
        "exposure_proxy": exposure,
    }


def extract_all_ped_features(
    targets: dict,
    cache_dir: str = "data/ped",
) -> list[dict]:
    """Extract PED structural features for all mutations.

    Uses WT ensemble structure to characterize the local environment
    at each mutation site.
    """
    from brain_idp_flow.data.ped_loader import load_ped_or_fallback

    results = []

    for tid, target in targets.items():
        print(f"Extracting PED features for {target.name}...")
        ensemble = load_ped_or_fallback(
            target.ped_id, target.length, cache_dir=cache_dir
        )
        print(f"  Ensemble: {ensemble.shape}")

        for mut in target.mutations:
            pos_0 = mut.pos - 1  # convert to 0-indexed
            features = extract_mutation_site_features(ensemble, pos_0)
            results.append({
                "target": tid,
                "mutation": mut.id,
                "pos": mut.pos,
                "agg_rate": mut.agg_rate_relative,
                **features,
            })
            print(f"  {mut.id}: RMSF={features['site_rmsf']:.2f}, "
                  f"CF={features['site_contact_freq']:.3f}, "
                  f"LR_CF={features['site_long_range_cf']:.3f}")

    return results
