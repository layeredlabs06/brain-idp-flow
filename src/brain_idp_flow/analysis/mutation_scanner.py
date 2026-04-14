"""Exhaustive single-point mutation landscape scanning.

Two-pass strategy:
1. Fast ESM-2 LLR scan for all 20×L possible mutations
2. Targeted flow model sampling for top-N candidates
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from brain_idp_flow.analysis.esm2_llr import ESM2MutationScorer
from brain_idp_flow.analysis.trajectory_analysis import extract_trajectory_features

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def scan_esm2_landscape(
    sequence: str,
    device: Optional[torch.device] = None,
    model_name: str = "esm2_t12_35M_UR50D",
) -> dict:
    """Pass 1: Score all possible single-point mutations via ESM-2 LLR.

    Args:
        sequence: wild-type amino acid sequence
        device: compute device
        model_name: ESM-2 model to use

    Returns:
        dict with:
            "llr_matrix": (20, L) LLR scores per AA per position
            "aa_order": str of 20 AAs (row order)
            "top_mutations": list of (pos_1indexed, wt, mt, llr) sorted by |llr|
    """
    scorer = ESM2MutationScorer(model_name=model_name, device=device)
    L = len(sequence)
    llr_matrix = np.full((20, L), np.nan)

    all_mutations: list[dict] = []

    for pos_0 in range(L):
        wt_aa = sequence[pos_0]
        pos_1 = pos_0 + 1

        for aa_idx, mt_aa in enumerate(STANDARD_AA):
            if mt_aa == wt_aa:
                llr_matrix[aa_idx, pos_0] = 0.0
                continue

            result = scorer.score_mutation(
                sequence, pos_1, wt_aa, mt_aa, fast=True,
            )
            llr = result["llr_site"]
            llr_matrix[aa_idx, pos_0] = llr
            all_mutations.append({
                "pos": pos_1,
                "pos_0": pos_0,
                "wt": wt_aa,
                "mt": mt_aa,
                "llr_site": llr,
            })

    # Sort by |LLR| descending (most impactful mutations first)
    all_mutations.sort(key=lambda m: abs(m["llr_site"]), reverse=True)

    return {
        "llr_matrix": llr_matrix,
        "aa_order": STANDARD_AA,
        "all_mutations": all_mutations,
    }


def scan_flow_model(
    model,
    embedder,
    target,
    candidate_mutations: list[dict],
    n_ensemble: int = 32,
    n_trajectory: int = 4,
    n_steps: int = 50,
    device: Optional[torch.device] = None,
) -> list[dict]:
    """Pass 2: Score top candidate mutations with the flow model.

    Args:
        model: trained MutationConditionedStructureHead
        embedder: ESM2Embedder instance
        target: Target object with sequence info
        candidate_mutations: list of dicts from scan_esm2_landscape
        n_ensemble: ensemble samples per mutation
        n_trajectory: trajectory samples per mutation
        n_steps: ODE integration steps
        device: compute device

    Returns:
        list of dicts with ESM-2 + flow model features per mutation
    """
    from brain_idp_flow.sample import sample_ensemble_with_trajectory
    from brain_idp_flow.geometry.metrics import radius_of_gyration

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # WT reference ensemble + trajectory
    wt_emb = embedder.embed_single(target.sequence)
    wt_result = sample_ensemble_with_trajectory(
        model, wt_emb,
        mut_pos=0, mut_aa=0,
        n_samples=n_ensemble,
        n_trajectory_samples=n_trajectory,
        n_steps=n_steps,
        device=device,
    )
    wt_rg = radius_of_gyration(
        torch.from_numpy(wt_result["ensemble"])
    ).mean().item()

    results = []

    for i, mut in enumerate(candidate_mutations):
        # Build mutant sequence
        mut_seq = list(target.sequence)
        mut_seq[mut["pos_0"]] = mut["mt"]
        mut_seq = "".join(mut_seq)

        # ESM-2 embedding for mutant
        mut_emb = embedder.embed_single(mut_seq)

        # AA index for model conditioning
        from brain_idp_flow.data.dataset import ProteinEnsembleDataset
        aa_idx = ProteinEnsembleDataset.AA_TO_IDX.get(mut["mt"], 0)

        # Flow model ensemble + trajectory
        mut_result = sample_ensemble_with_trajectory(
            model, mut_emb,
            mut_pos=mut["pos"],
            mut_aa=aa_idx,
            n_samples=n_ensemble,
            n_trajectory_samples=n_trajectory,
            n_steps=n_steps,
            device=device,
        )

        # Delta Rg
        mut_rg = radius_of_gyration(
            torch.from_numpy(mut_result["ensemble"])
        ).mean().item()
        delta_rg = mut_rg - wt_rg

        # Trajectory features
        traj_feats = extract_trajectory_features(
            mut_result["trajectory"],
            mutation_pos=mut["pos_0"],
        )

        results.append({
            **mut,
            "delta_rg": delta_rg,
            "rg_mutant": mut_rg,
            "rg_wt": wt_rg,
            **{k: v for k, v in traj_feats.items() if not k.startswith("_")},
        })

        if (i + 1) % 10 == 0:
            print(f"  Scanned {i + 1}/{len(candidate_mutations)} mutations")

    return results


def scan_full_landscape(
    model,
    embedder,
    target,
    target_id: str,
    top_n: int = 50,
    n_ensemble: int = 32,
    n_trajectory: int = 4,
    n_steps: int = 50,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Full 2-pass mutation landscape scan for a single protein target.

    Args:
        model: trained flow model
        embedder: ESM2Embedder
        target: Target object
        target_id: e.g. "tau_K18"
        top_n: number of top ESM-2 candidates for flow model scan
        n_ensemble: ensemble size for flow model
        n_trajectory: trajectory samples for flow model
        n_steps: ODE steps
        device: compute device
        output_dir: save figures here (optional)

    Returns:
        dict with llr_matrix, top_flow_results, known_mutation_ranks
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== Scanning {target.name} ({len(target.sequence)} aa) ===")

    # Pass 1: ESM-2 LLR for all mutations
    print("Pass 1: ESM-2 LLR scanning...")
    esm2_scan = scan_esm2_landscape(target.sequence, device=device)
    print(f"  Scored {len(esm2_scan['all_mutations'])} mutations")

    # Check where known mutations rank
    known_ids = {(m.pos, m.mt) for m in target.mutations}
    known_ranks = []
    for rank, mut in enumerate(esm2_scan["all_mutations"]):
        if (mut["pos"], mut["mt"]) in known_ids:
            known_ranks.append({
                "mutation": f"{mut['wt']}{mut['pos']}{mut['mt']}",
                "rank": rank + 1,
                "percentile": (rank + 1) / len(esm2_scan["all_mutations"]) * 100,
                "llr": mut["llr_site"],
            })

    print("  Known mutation ranks:")
    for kr in known_ranks:
        print(f"    {kr['mutation']}: rank {kr['rank']} "
              f"(top {kr['percentile']:.1f}%)")

    # Pass 2: Flow model for top candidates
    top_candidates = esm2_scan["all_mutations"][:top_n]
    print(f"\nPass 2: Flow model scanning top {top_n} candidates...")
    flow_results = scan_flow_model(
        model, embedder, target, top_candidates,
        n_ensemble=n_ensemble,
        n_trajectory=n_trajectory,
        n_steps=n_steps,
        device=device,
    )

    # Generate heatmap
    if output_dir:
        _plot_landscape_heatmap(
            esm2_scan["llr_matrix"],
            target.sequence,
            target.name,
            target.mutations,
            Path(output_dir) / f"{target_id}_landscape.png",
        )

    return {
        "target_id": target_id,
        "esm2_scan": esm2_scan,
        "flow_results": flow_results,
        "known_ranks": known_ranks,
    }


def _plot_landscape_heatmap(
    llr_matrix: np.ndarray,
    sequence: str,
    target_name: str,
    known_mutations: list,
    save_path: Path,
) -> None:
    """Plot mutation landscape heatmap with known mutations marked."""
    fig, ax = plt.subplots(figsize=(max(12, len(sequence) * 0.15), 6))

    # Clip for visualization
    vmax = min(np.nanpercentile(np.abs(llr_matrix), 99), 5.0)
    im = ax.imshow(
        llr_matrix, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )

    # Mark known pathogenic mutations
    for mut in known_mutations:
        pos_0 = mut.pos - 1
        aa_idx = STANDARD_AA.index(mut.mt) if mut.mt in STANDARD_AA else -1
        if aa_idx >= 0:
            ax.plot(pos_0, aa_idx, "k*", markersize=8)

    ax.set_yticks(range(20))
    ax.set_yticklabels(list(STANDARD_AA), fontsize=8)
    ax.set_xlabel("Sequence Position", fontsize=11)
    ax.set_ylabel("Mutant Amino Acid", fontsize=11)
    ax.set_title(
        f"{target_name} — ESM-2 LLR Mutation Landscape\n"
        f"(* = known pathogenic mutations)",
        fontsize=12,
    )

    fig.colorbar(im, ax=ax, label="Log-Likelihood Ratio", shrink=0.8)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)
