"""Gradio web app for hybrid aggregation prediction.

Usage:
    python -m brain_idp_flow.app
    # or: gradio src/brain_idp_flow/app.py

Inputs: protein sequence + mutation (e.g. "E22G")
Outputs: aggregation risk score, feature contribution chart, Rg distribution plot
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Lazy globals — loaded once on first prediction
_embedder = None
_flow_model = None
_scorer = None
_predictor = None
_config = None
_device = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_embedder():
    global _embedder
    if _embedder is None:
        from brain_idp_flow.features.seq_embed import ESM2Embedder
        _embedder = ESM2Embedder(
            model_name="esm2_t12_35M_UR50D",
            device=_get_device(),
        )
    return _embedder


def _load_scorer():
    global _scorer
    if _scorer is None:
        from brain_idp_flow.analysis.esm2_llr import ESM2MutationScorer
        _scorer = ESM2MutationScorer(device=_get_device())
    return _scorer


def _load_flow_model():
    global _flow_model, _config
    if _flow_model is None:
        import yaml
        from brain_idp_flow.sample import load_model

        config_path = os.environ.get("FLOW_CONFIG", "configs/flow.yaml")
        ckpt_path = os.environ.get("FLOW_CKPT", "runs/flow_35m/best.pt")

        _config = yaml.safe_load(open(config_path))
        _flow_model = load_model(_config, ckpt_path, _get_device())
    return _flow_model


def _load_predictor():
    global _predictor
    if _predictor is None:
        import pickle

        model_path = os.environ.get("HYBRID_MODEL", "runs/hybrid_predictor.pkl")
        if Path(model_path).exists():
            with open(model_path, "rb") as f:
                _predictor = pickle.load(f)
        else:
            from brain_idp_flow.model.hybrid_predictor import HybridAggregationPredictor
            _predictor = HybridAggregationPredictor()
            print(f"WARNING: No pre-trained hybrid model at {model_path}. "
                  f"Predictions will fail until fit() is called.")
    return _predictor


def _parse_mutation(mutation_str: str) -> tuple[str, int, str]:
    """Parse mutation string like 'E22G' -> (wt='E', pos=22, mt='G')."""
    mutation_str = mutation_str.strip().upper()
    if len(mutation_str) < 3:
        raise ValueError(f"Invalid mutation format: '{mutation_str}'. Expected e.g. 'E22G'.")

    wt = mutation_str[0]
    mt = mutation_str[-1]
    pos_str = mutation_str[1:-1]

    if not wt.isalpha() or not mt.isalpha():
        raise ValueError(f"Invalid amino acids in '{mutation_str}'.")
    try:
        pos = int(pos_str)
    except ValueError:
        raise ValueError(f"Invalid position in '{mutation_str}'.")

    return wt, pos, mt


def predict(sequence: str, mutation: str) -> tuple:
    """Main prediction function for Gradio interface.

    Args:
        sequence: protein amino acid sequence
        mutation: mutation string (e.g., "E22G")

    Returns:
        (score_text, contribution_fig, rg_fig)
    """
    try:
        wt_aa, pos, mt_aa = _parse_mutation(mutation)
    except ValueError as e:
        return str(e), None, None

    # Validate position
    if pos < 1 or pos > len(sequence):
        return f"Position {pos} out of range (sequence length: {len(sequence)})", None, None
    if sequence[pos - 1] != wt_aa:
        return (
            f"WT amino acid mismatch: expected '{wt_aa}' at position {pos}, "
            f"found '{sequence[pos - 1]}'",
            None, None,
        )

    device = _get_device()
    embedder = _load_embedder()
    scorer = _load_scorer()
    flow_model = _load_flow_model()
    predictor = _load_predictor()

    # 1. ESM-2 embedding (mean-pooled)
    mut_seq = list(sequence)
    mut_seq[pos - 1] = mt_aa
    mut_seq_str = "".join(mut_seq)

    mut_emb_full = embedder.embed_single(mut_seq_str)  # (L, D)
    mut_emb_pooled = mut_emb_full.mean(dim=0).cpu().numpy()  # (D,)

    # 2. ESM-2 LLR
    llr_result = scorer.score_mutation(sequence, pos, wt_aa, mt_aa, fast=True)
    llr_value = llr_result["llr_site"]

    # 3. Flow ensembles
    from brain_idp_flow.sample import sample_ensemble
    from brain_idp_flow.model.hybrid_predictor import EnsembleFeatureExtractor

    wt_emb = embedder.embed_single(sequence)
    AA_TO_IDX = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    mut_aa_idx = AA_TO_IDX.get(mt_aa, 0)

    wt_ensemble = sample_ensemble(
        flow_model, wt_emb, mut_pos=0, mut_aa=0,
        n_samples=100, n_steps=50, device=device, batch_size=32,
    )
    mut_ensemble = sample_ensemble(
        flow_model, mut_emb_full, mut_pos=pos, mut_aa=mut_aa_idx,
        n_samples=100, n_steps=50, device=device, batch_size=32,
    )

    # 4. Structural features
    extractor = EnsembleFeatureExtractor()
    struct_feats = extractor.extract(wt_ensemble, mut_ensemble, pos)

    # 5. Predict
    if not predictor.is_fitted:
        score_text = (
            f"**Hybrid model not trained yet.**\n\n"
            f"Structural features extracted:\n"
            + "\n".join(f"- {k}: {v:.4f}" for k, v in struct_feats.items())
            + f"\n- LLR: {llr_value:.4f}"
        )
        score = 0.0
    else:
        score = predictor.predict_single(mut_emb_pooled, struct_feats, llr_value)
        risk = "HIGH" if score > 0.5 else "MEDIUM" if score > 0 else "LOW"
        score_text = (
            f"## Aggregation Risk: **{risk}**\n\n"
            f"Score: **{score:.3f}**\n\n"
            f"| Feature | Value |\n|---|---|\n"
            + "\n".join(f"| {k} | {v:.4f} |" for k, v in struct_feats.items())
            + f"\n| LLR | {llr_value:.4f} |"
        )

    # 6. Feature contribution chart
    contribution_fig = _plot_contributions(
        predictor, mut_emb_pooled, struct_feats, llr_value, mutation,
    )

    # 7. Rg distribution comparison
    from brain_idp_flow.geometry.metrics import radius_of_gyration
    rg_wt = radius_of_gyration(torch.from_numpy(wt_ensemble)).numpy()
    rg_mut = radius_of_gyration(torch.from_numpy(mut_ensemble)).numpy()
    rg_fig = _plot_rg_distributions(rg_wt, rg_mut, mutation)

    return score_text, contribution_fig, rg_fig


def _plot_contributions(
    predictor,
    embedding: np.ndarray,
    struct_feats: dict,
    llr_value: float,
    mutation: str,
) -> Optional[plt.Figure]:
    """Plot feature contributions bar chart."""
    if not predictor.is_fitted:
        return None

    contributions = predictor.explain(embedding, struct_feats, llr_value)
    top = contributions[:12]

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [c.name for c in top]
    vals = [c.contribution for c in top]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]

    ax.barh(range(len(top)), vals, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Contribution")
    ax.set_title(f"Top Feature Contributions: {mutation}")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def _plot_rg_distributions(
    rg_wt: np.ndarray,
    rg_mut: np.ndarray,
    mutation: str,
) -> plt.Figure:
    """Plot WT vs mutant Rg distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rg_wt, bins=25, alpha=0.5, density=True,
            label=f"WT (mean={rg_wt.mean():.1f} A)", color="#4575b4")
    ax.hist(rg_mut, bins=25, alpha=0.5, density=True,
            label=f"{mutation} (mean={rg_mut.mean():.1f} A)", color="#d73027")
    ax.axvline(rg_wt.mean(), color="#4575b4", linestyle="--", alpha=0.7)
    ax.axvline(rg_mut.mean(), color="#d73027", linestyle="--", alpha=0.7)
    ax.set_xlabel("Radius of Gyration (A)")
    ax.set_ylabel("Density")
    ax.set_title(f"Rg Distribution: WT vs {mutation}")
    ax.legend()
    fig.tight_layout()
    return fig


def create_app():
    """Create and return the Gradio interface."""
    import gradio as gr

    from brain_idp_flow.data.dms_loader import ABETA42_WT

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(
                label="Protein Sequence",
                placeholder="Enter amino acid sequence...",
                value=ABETA42_WT,
                lines=3,
            ),
            gr.Textbox(
                label="Mutation",
                placeholder="e.g. E22G",
                value="E22G",
            ),
        ],
        outputs=[
            gr.Markdown(label="Prediction"),
            gr.Plot(label="Feature Contributions"),
            gr.Plot(label="Rg Distribution"),
        ],
        title="Brain IDP Aggregation Predictor",
        description=(
            "Hybrid model combining ESM-2 embeddings, flow-model structural features, "
            "and evolutionary constraints (LLR) to predict mutation effects on IDP aggregation."
        ),
        examples=[
            [ABETA42_WT, "E22G"],
            [ABETA42_WT, "E22Q"],
            [ABETA42_WT, "D23N"],
            [ABETA42_WT, "A21G"],
        ],
    )
    return demo


def main():
    demo = create_app()
    demo.launch(share=False)


if __name__ == "__main__":
    main()
