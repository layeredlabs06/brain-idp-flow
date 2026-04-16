"""Gradio web app for IDP aggregation prediction.

ESM-2 embedding-only predictor. No flow model needed.

Usage:
    python -m brain_idp_flow.app
    # or with custom model:
    MODEL_PATH=runs/embedding_predictor.pkl python -m brain_idp_flow.app

Inputs: protein sequence + mutation (e.g. "E22G")
Outputs: aggregation risk score, PCA feature importance chart
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

# Lazy globals
_embedder = None
_predictor = None
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


def _load_predictor():
    global _predictor
    if _predictor is None:
        from brain_idp_flow.model.embedding_predictor import EmbeddingAggregationPredictor

        model_path = os.environ.get("MODEL_PATH", "runs/embedding_predictor.pkl")
        if Path(model_path).exists():
            _predictor = EmbeddingAggregationPredictor.load(model_path)
            print(f"Loaded model from {model_path}")
        else:
            _predictor = EmbeddingAggregationPredictor()
            print(f"WARNING: No model at {model_path}. Run training first.")
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

    Returns:
        (score_markdown, importance_figure)
    """
    try:
        wt_aa, pos, mt_aa = _parse_mutation(mutation)
    except ValueError as e:
        return str(e), None

    if pos < 1 or pos > len(sequence):
        return f"Position {pos} out of range (sequence length: {len(sequence)})", None
    if sequence[pos - 1] != wt_aa:
        return (
            f"WT amino acid mismatch: expected '{wt_aa}' at position {pos}, "
            f"found '{sequence[pos - 1]}'",
            None,
        )

    embedder = _load_embedder()
    predictor = _load_predictor()

    # Compute mutant embedding (mean-pooled)
    mut_seq = list(sequence)
    mut_seq[pos - 1] = mt_aa
    mut_seq_str = "".join(mut_seq)

    emb = embedder.embed_single(mut_seq_str)  # (L, D)
    emb_pooled = emb.mean(dim=0).cpu().numpy()  # (D,)

    if not predictor.is_fitted:
        return "**Model not trained.** Run `colab/11_hybrid_model.ipynb` first.", None

    result = predictor.predict_single(emb_pooled)

    # Risk color
    color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}[result.risk_level]

    score_text = (
        f"## Aggregation Risk: **<span style='color:{color}'>{result.risk_level}</span>**\n\n"
        f"**Score: {result.score:.3f}**\n\n"
        f"Positive = more aggregation-prone than WT\n\n"
        f"---\n\n"
        f"*Prediction based on ESM-2 35M embeddings (PCA-50, GBRegressor)*\n\n"
        f"*CV performance on Aβ42 DMS: ρ=0.595 [0.543, 0.641]*"
    )

    # Feature importance chart
    fig = _plot_importance(result.top_features, mutation)

    return score_text, fig


def _plot_importance(
    top_features: tuple[tuple[str, float], ...],
    mutation: str,
) -> plt.Figure:
    """Plot PCA component importance."""
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [f[0] for f in top_features]
    vals = [f[1] for f in top_features]

    ax.barh(range(len(names)), vals, color="#d73027", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Importance (GBRegressor)")
    ax.set_title(f"Top PCA Components: {mutation}")
    ax.invert_yaxis()
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
            gr.Plot(label="Feature Importance"),
        ],
        title="IDP Aggregation Predictor",
        description=(
            "Predict mutation effects on IDP aggregation using ESM-2 embeddings. "
            "No structure generation needed. ~1 second per prediction."
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
