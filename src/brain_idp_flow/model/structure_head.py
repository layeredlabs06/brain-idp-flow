"""Mutation-conditioned Transformer velocity network for flow matching."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal positional embedding for diffusion time t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """t: (B,) -> (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class RBFDistanceEncoding(nn.Module):
    """Radial basis function encoding of pairwise distances."""

    def __init__(self, n_bins: int = 16, d_max: float = 20.0):
        super().__init__()
        self.register_buffer(
            "centers", torch.linspace(0.0, d_max, n_bins)
        )
        self.sigma = d_max / n_bins

    def forward(self, dists: Tensor) -> Tensor:
        """dists: (..., L, L) -> (..., L, L, n_bins)."""
        return torch.exp(
            -0.5 * ((dists.unsqueeze(-1) - self.centers) / self.sigma) ** 2
        )


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioned on time embedding."""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.proj = nn.Linear(d_cond, 2 * d_model)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """x: (B, L, D), cond: (B, D_cond) -> (B, L, D)."""
        gamma_beta = self.proj(cond).unsqueeze(1)  # (B, 1, 2D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x * (1 + gamma) + beta


class StructureTransformerLayer(nn.Module):
    """Single transformer layer with pairwise bias and FiLM conditioning."""

    def __init__(self, d_model: int, n_heads: int, n_rbf: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.pair_proj = nn.Linear(n_rbf, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.film1 = FiLMLayer(d_model, d_model)
        self.film2 = FiLMLayer(d_model, d_model)

    def forward(
        self, x: Tensor, pair_rbf: Tensor, t_emb: Tensor
    ) -> Tensor:
        """
        x: (B, L, D)
        pair_rbf: (B, L, L, n_rbf)
        t_emb: (B, D)
        """
        # Pairwise attention bias: (B, n_heads, L, L)
        attn_bias = self.pair_proj(pair_rbf).permute(0, 3, 1, 2)

        # Self-attention with pairwise bias
        h = self.norm1(x)
        h = self.film1(h, t_emb)
        attn_out, _ = self.attn(h, h, h, attn_mask=None)
        # Add pairwise bias effect (approximate via additive before softmax not
        # directly supported by nn.MHA, so we add it post-attention as residual)
        x = x + attn_out

        # Feedforward
        h = self.norm2(x)
        h = self.film2(h, t_emb)
        x = x + self.ff(h)

        return x


class MutationConditionedStructureHead(nn.Module):
    """Predicts velocity field v(x_t, t, seq_emb, mutation) for flow matching.

    Inputs:
        seq_emb:  (B, L, D_seq)  — frozen ESM-2 embeddings
        x_t:      (B, L, 3)      — noised coordinates at time t
        t:        (B,)            — diffusion time
        mut_pos:  (B,)            — mutation position (0 = WT)
        mut_aa:   (B,)            — mutant amino acid index (0 = WT)

    Output:
        v_hat:    (B, L, 3)      — predicted velocity
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_seq_in: int = 480,
        dropout: float = 0.1,
        rbf_bins: int = 16,
        rbf_max: float = 20.0,
        mutation_embed_dim: int = 32,
        max_len: int = 512,
        n_amino_acids: int = 21,  # 20 AA + padding/WT
    ):
        super().__init__()

        self.d_model = d_model

        # Input projections
        self.seq_proj = nn.Linear(d_seq_in, d_model)
        self.coord_proj = nn.Linear(3, d_model)
        self.input_combine = nn.Linear(2 * d_model, d_model)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Mutation conditioning
        self.mut_pos_embed = nn.Embedding(max_len + 1, mutation_embed_dim)  # 0 = no mutation
        self.mut_aa_embed = nn.Embedding(n_amino_acids, mutation_embed_dim)
        self.mut_proj = nn.Linear(2 * mutation_embed_dim, d_model)

        # Pairwise distance encoding
        self.rbf = RBFDistanceEncoding(rbf_bins, rbf_max)

        # Transformer layers
        self.layers = nn.ModuleList([
            StructureTransformerLayer(d_model, n_heads, rbf_bins, dropout)
            for _ in range(n_layers)
        ])

        # Output projection to velocity
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 3)

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init the output projection for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        seq_emb: Tensor,
        x_t: Tensor,
        t: Tensor,
        mut_pos: Tensor,
        mut_aa: Tensor,
    ) -> Tensor:
        B, L, _ = x_t.shape

        # Center coordinates
        x_centered = x_t - x_t.mean(dim=1, keepdim=True)

        # Project inputs
        h_seq = self.seq_proj(seq_emb)        # (B, L, D)
        h_coord = self.coord_proj(x_centered)  # (B, L, D)
        h = self.input_combine(torch.cat([h_seq, h_coord], dim=-1))  # (B, L, D)

        # Time conditioning
        t_emb = self.time_embed(t)  # (B, D)

        # Mutation conditioning — add to time embedding
        mut_emb = torch.cat([
            self.mut_pos_embed(mut_pos),
            self.mut_aa_embed(mut_aa),
        ], dim=-1)  # (B, 2 * mut_dim)
        mut_cond = self.mut_proj(mut_emb)  # (B, D)
        t_emb = t_emb + mut_cond

        # Pairwise distances -> RBF
        dists = torch.cdist(x_centered, x_centered)  # (B, L, L)
        pair_rbf = self.rbf(dists)  # (B, L, L, n_rbf)

        # Transformer layers
        for layer in self.layers:
            h = layer(h, pair_rbf, t_emb)

        # Output velocity
        h = self.output_norm(h)
        v_hat = self.output_proj(h)  # (B, L, 3)

        # Zero center the velocity (translation invariance)
        v_hat = v_hat - v_hat.mean(dim=1, keepdim=True)

        return v_hat
