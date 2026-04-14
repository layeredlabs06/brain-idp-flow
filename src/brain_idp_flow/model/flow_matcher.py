"""Conditional Flow Matching loss and ODE sampler (Lipman et al., ICLR 2023)."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


class ConditionalFlowMatcher(nn.Module):
    """Conditional flow matching training objective.

    Given ground truth x_1 and noise x_0 ~ N(0, I):
        x_t = (1 - t) * x_0 + t * x_1
        target velocity u = x_1 - x_0
        loss = MSE(v_hat(x_t, t), u)
    """

    def compute_loss(
        self,
        model_fn: Callable[..., Tensor],
        x_1: Tensor,
        seq_emb: Tensor,
        mut_pos: Tensor,
        mut_aa: Tensor,
    ) -> Tensor:
        """Compute FM loss for a batch.

        Args:
            model_fn: velocity network (seq_emb, x_t, t, mut_pos, mut_aa) -> v_hat
            x_1: (B, L, 3) ground truth Cα coords (centered)
            seq_emb: (B, L, D) frozen sequence embeddings
            mut_pos: (B,) mutation position
            mut_aa: (B,) mutant amino acid index

        Returns:
            scalar loss
        """
        B, L, _ = x_1.shape
        device = x_1.device

        # Sample time uniformly
        t = torch.rand(B, device=device)

        # Sample noise
        x_0 = torch.randn_like(x_1)
        x_0 = x_0 - x_0.mean(dim=1, keepdim=True)  # center noise

        # Interpolate
        t_expand = t[:, None, None]  # (B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # Target velocity
        u = x_1 - x_0

        # Predict velocity
        v_hat = model_fn(seq_emb, x_t, t, mut_pos, mut_aa)

        # MSE loss
        return ((v_hat - u) ** 2).mean()


class ODESampler:
    """ODE sampler for generating structures from noise.

    Integrates: dx/dt = v(x_t, t) from t=0 to t=1.
    """

    def __init__(
        self,
        model_fn: Callable[..., Tensor],
        n_steps: int = 50,
        method: str = "euler",
    ):
        self.model_fn = model_fn
        self.n_steps = n_steps
        self.method = method

    @torch.no_grad()
    def sample(
        self,
        seq_emb: Tensor,
        mut_pos: Tensor,
        mut_aa: Tensor,
        n_samples: int = 1,
        return_trajectory: bool = False,
    ) -> Tensor | dict:
        """Generate structure samples via ODE integration.

        Args:
            seq_emb: (1, L, D) or (B, L, D) sequence embeddings
            mut_pos: (B,) mutation position
            mut_aa: (B,) mutant amino acid index
            n_samples: number of samples per input (if seq_emb is single)
            return_trajectory: if True, return full trajectory with velocities

        Returns:
            If return_trajectory is False: (B * n_samples, L, 3) Cα coordinates
            If return_trajectory is True: dict with keys:
                "final": (B, L, 3) final coordinates
                "coords": (n_steps, B, L, 3) coordinates at each step
                "velocities": (n_steps, B, L, 3) velocity at each step
                "times": (n_steps,) time values
        """
        if seq_emb.shape[0] == 1 and n_samples > 1:
            seq_emb = seq_emb.expand(n_samples, -1, -1)
            mut_pos = mut_pos.expand(n_samples)
            mut_aa = mut_aa.expand(n_samples)

        B, L, _ = seq_emb.shape
        device = seq_emb.device

        # Start from noise
        x = torch.randn(B, L, 3, device=device)
        x = x - x.mean(dim=1, keepdim=True)

        dt = 1.0 / self.n_steps

        # Trajectory storage
        if return_trajectory:
            coords_traj: list[Tensor] = []
            vel_traj: list[Tensor] = []
            time_traj: list[float] = []

        for step in range(self.n_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device)

            if self.method == "euler":
                v = self.model_fn(seq_emb, x, t, mut_pos, mut_aa)
                if return_trajectory:
                    coords_traj.append(x.clone())
                    vel_traj.append(v.clone())
                    time_traj.append(t_val)
                x = x + v * dt

            elif self.method == "heun":
                v1 = self.model_fn(seq_emb, x, t, mut_pos, mut_aa)
                x_pred = x + v1 * dt

                t_next = torch.full((B,), min(t_val + dt, 1.0), device=device)
                v2 = self.model_fn(seq_emb, x_pred, t_next, mut_pos, mut_aa)
                if return_trajectory:
                    coords_traj.append(x.clone())
                    vel_traj.append(0.5 * (v1 + v2))
                    time_traj.append(t_val)
                x = x + 0.5 * (v1 + v2) * dt

            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Re-center at each step
            x = x - x.mean(dim=1, keepdim=True)

        if return_trajectory:
            return {
                "final": x,
                "coords": torch.stack(coords_traj),
                "velocities": torch.stack(vel_traj),
                "times": torch.tensor(time_traj),
            }
        return x
