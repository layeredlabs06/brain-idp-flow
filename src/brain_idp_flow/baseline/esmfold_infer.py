"""Phase 1 baseline: ESMFold pseudo-ensemble generation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def generate_esmfold_ensemble(
    sequence: str,
    n_samples: int = 50,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate pseudo-ensemble using ESMFold with dropout perturbation.

    Args:
        sequence: amino acid string
        n_samples: number of structures to generate
        device: compute device (default: cuda if available)

    Returns:
        coords: (n_samples, L, 3) Cα coordinates
        plddt: (n_samples, L) per-residue pLDDT scores
    """
    import esm

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = esm.pretrained.esmfold_v1()
    model = model.to(device)

    all_coords = []
    all_plddt = []

    for i in range(n_samples):
        # Set seed for reproducibility
        torch.manual_seed(i * 42)

        if i == 0:
            # First sample: deterministic (eval mode)
            model.eval()
        else:
            # Enable dropout for diversity
            model.train()
            # But keep batchnorm in eval mode
            for module in model.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.eval()

        with torch.no_grad():
            output = model.infer(sequence)

        # Extract Cα positions (atom index 1 in atom37 representation)
        positions = output["positions"][-1]  # last layer, (1, L, 37, 3)
        ca_coords = positions[0, :, 1, :].cpu().numpy()  # Cα is index 1
        plddt = output["plddt"][0].cpu().numpy()  # (L,)

        all_coords.append(ca_coords)
        all_plddt.append(plddt)

    model.eval()  # restore

    return np.stack(all_coords), np.stack(all_plddt)
