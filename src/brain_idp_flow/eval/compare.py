"""Compare generated ensembles against PED reference and baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from brain_idp_flow.geometry.metrics import (
    radius_of_gyration,
    contact_frequency,
    end_to_end_distance,
    js_divergence_1d,
)


@dataclass
class EnsembleMetrics:
    rg_mean: float
    rg_std: float
    rg_js_vs_ref: float
    e2e_mean: float
    e2e_std: float
    e2e_js_vs_ref: float
    contact_map_l1: float
    n_samples: int

    def summary(self) -> str:
        return (
            f"Rg={self.rg_mean:.2f}±{self.rg_std:.2f} "
            f"(JS={self.rg_js_vs_ref:.4f})  "
            f"E2E={self.e2e_mean:.2f}±{self.e2e_std:.2f} "
            f"(JS={self.e2e_js_vs_ref:.4f})  "
            f"ContactL1={self.contact_map_l1:.4f}"
        )


def compare_ensembles(
    generated: np.ndarray,
    reference: np.ndarray,
    label: str = "",
) -> EnsembleMetrics:
    """Compare a generated ensemble against a reference ensemble.

    Both arrays: (K, L, 3) Cα coordinates.
    """
    gen_t = torch.from_numpy(generated)
    ref_t = torch.from_numpy(reference)

    # Truncate to common length
    min_len = min(gen_t.shape[1], ref_t.shape[1])
    gen_t = gen_t[:, :min_len, :]
    ref_t = ref_t[:, :min_len, :]

    # Radius of gyration
    rg_gen = radius_of_gyration(gen_t).numpy()
    rg_ref = radius_of_gyration(ref_t).numpy()
    rg_js = js_divergence_1d(rg_gen, rg_ref)

    # End-to-end distance
    e2e_gen = end_to_end_distance(gen_t).numpy()
    e2e_ref = end_to_end_distance(ref_t).numpy()
    e2e_js = js_divergence_1d(e2e_gen, e2e_ref)

    # Contact map frequency
    cf_gen = contact_frequency(gen_t).numpy()
    cf_ref = contact_frequency(ref_t).numpy()
    contact_l1 = float(np.abs(cf_gen - cf_ref).mean())

    return EnsembleMetrics(
        rg_mean=float(rg_gen.mean()),
        rg_std=float(rg_gen.std()),
        rg_js_vs_ref=rg_js,
        e2e_mean=float(e2e_gen.mean()),
        e2e_std=float(e2e_gen.std()),
        e2e_js_vs_ref=e2e_js,
        contact_map_l1=contact_l1,
        n_samples=len(generated),
    )


def compare_mutation_effect(
    wt_ensemble: np.ndarray,
    mut_ensemble: np.ndarray,
    reference_wt: np.ndarray,
) -> dict:
    """Compare WT vs mutant ensembles to quantify mutation effect.

    Returns dict with delta metrics and aggregation-related scores.
    """
    wt_t = torch.from_numpy(wt_ensemble)
    mut_t = torch.from_numpy(mut_ensemble)

    min_len = min(wt_t.shape[1], mut_t.shape[1])
    wt_t = wt_t[:, :min_len, :]
    mut_t = mut_t[:, :min_len, :]

    rg_wt = radius_of_gyration(wt_t).numpy()
    rg_mut = radius_of_gyration(mut_t).numpy()

    e2e_wt = end_to_end_distance(wt_t).numpy()
    e2e_mut = end_to_end_distance(mut_t).numpy()

    cf_wt = contact_frequency(wt_t).numpy()
    cf_mut = contact_frequency(mut_t).numpy()

    # Aggregation proxy: increased exposed hydrophobic surface
    # approximated by decreased contact frequency (more open conformations)
    contact_change = float((cf_mut - cf_wt).mean())

    return {
        "delta_rg_mean": float(rg_mut.mean() - rg_wt.mean()),
        "delta_e2e_mean": float(e2e_mut.mean() - e2e_wt.mean()),
        "rg_js_wt_vs_mut": js_divergence_1d(rg_wt, rg_mut),
        "contact_change_mean": contact_change,
        "rg_wt": float(rg_wt.mean()),
        "rg_mut": float(rg_mut.mean()),
    }
