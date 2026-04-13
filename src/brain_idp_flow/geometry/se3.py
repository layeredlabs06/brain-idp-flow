"""SE(3) utilities: Kabsch alignment, random rotations, centering."""

from __future__ import annotations

import torch
from torch import Tensor


def center(coords: Tensor) -> Tensor:
    """Remove center of mass. coords: (..., L, 3)."""
    return coords - coords.mean(dim=-2, keepdim=True)


def kabsch_align(mobile: Tensor, target: Tensor) -> Tensor:
    """Align *mobile* onto *target* using Kabsch algorithm.

    Both tensors: (..., L, 3).  Returns aligned mobile (new tensor).
    """
    mobile_c = center(mobile)
    target_c = center(target)

    # Cross-covariance  (..., 3, 3)
    H = torch.einsum("...li,...lj->...ij", mobile_c, target_c)
    U, _S, Vt = torch.linalg.svd(H)

    # Correct reflection
    d = torch.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2))
    sign = torch.ones_like(d)
    sign[d < 0] = -1.0

    # Build corrected V
    Vt_corr = Vt.clone()
    Vt_corr[..., -1, :] *= sign.unsqueeze(-1)

    R = Vt_corr.transpose(-1, -2) @ U.transpose(-1, -2)
    aligned = mobile_c @ R.transpose(-1, -2) + target.mean(dim=-2, keepdim=True)
    return aligned


def random_rotation(batch_size: int, device: torch.device) -> Tensor:
    """Sample uniform random rotation matrices via QR decomposition.

    Returns: (batch_size, 3, 3).
    """
    z = torch.randn(batch_size, 3, 3, device=device)
    Q, R = torch.linalg.qr(z)
    # Ensure proper rotation (det = +1)
    sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    Q = Q * sign.unsqueeze(-2)
    det = torch.det(Q)
    Q[det < 0, :, -1] *= -1
    return Q


def apply_random_rotation(coords: Tensor) -> Tensor:
    """Apply random SO(3) rotation to centered coords. coords: (B, L, 3)."""
    coords_c = center(coords)
    R = random_rotation(coords_c.shape[0], coords_c.device)
    return torch.einsum("bij,blj->bli", R, coords_c)


def rmsd(a: Tensor, b: Tensor) -> Tensor:
    """Per-sample RMSD after centering. a, b: (B, L, 3) -> (B,)."""
    diff = center(a) - center(b)
    return torch.sqrt((diff ** 2).sum(dim=(-1, -2)) / a.shape[-2])
