"""Sample ensembles from a trained flow matching model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from brain_idp_flow.model.structure_head import MutationConditionedStructureHead
from brain_idp_flow.model.flow_matcher import ODESampler
from brain_idp_flow.train import create_model


def load_model(
    config: dict,
    ckpt_path: str | Path,
    device: torch.device,
) -> MutationConditionedStructureHead:
    model = create_model(config).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def sample_ensemble(
    model: MutationConditionedStructureHead,
    seq_emb: Tensor,
    mut_pos: int = 0,
    mut_aa: int = 0,
    n_samples: int = 200,
    n_steps: int = 50,
    method: str = "euler",
    device: torch.device = torch.device("cpu"),
    batch_size: int = 32,
) -> np.ndarray:
    """Generate an ensemble of structures.

    Args:
        model: trained velocity network
        seq_emb: (L, D) sequence embedding
        mut_pos: mutation position (0 = WT)
        mut_aa: mutant AA index (0 = WT)
        n_samples: total samples to generate
        n_steps: ODE integration steps
        method: "euler" or "heun"
        device: compute device
        batch_size: samples per batch

    Returns:
        (n_samples, L, 3) Cα coordinates
    """
    sampler = ODESampler(model.forward, n_steps=n_steps, method=method)
    all_samples = []

    seq_emb_dev = seq_emb.unsqueeze(0).to(device)

    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        mp = torch.full((1,), mut_pos, dtype=torch.long, device=device)
        ma = torch.full((1,), mut_aa, dtype=torch.long, device=device)

        coords = sampler.sample(seq_emb_dev, mp, ma, n_samples=bs)
        all_samples.append(coords.cpu().numpy())
        remaining -= bs

    return np.concatenate(all_samples, axis=0)[:n_samples]
