"""Tests for flow matching model and loss."""

import torch
import pytest

from brain_idp_flow.model.structure_head import MutationConditionedStructureHead
from brain_idp_flow.model.flow_matcher import ConditionalFlowMatcher, ODESampler


@pytest.fixture
def small_model():
    return MutationConditionedStructureHead(
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_seq_in=16,
        dropout=0.0,
        rbf_bins=8,
        rbf_max=20.0,
        mutation_embed_dim=8,
    )


class TestStructureHead:
    def test_output_shape(self, small_model):
        B, L, D = 2, 10, 16
        seq_emb = torch.randn(B, L, D)
        x_t = torch.randn(B, L, 3)
        t = torch.rand(B)
        mut_pos = torch.zeros(B, dtype=torch.long)
        mut_aa = torch.zeros(B, dtype=torch.long)

        v = small_model(seq_emb, x_t, t, mut_pos, mut_aa)
        assert v.shape == (B, L, 3)

    def test_velocity_centered(self, small_model):
        B, L, D = 3, 15, 16
        seq_emb = torch.randn(B, L, D)
        x_t = torch.randn(B, L, 3)
        t = torch.rand(B)
        mut_pos = torch.zeros(B, dtype=torch.long)
        mut_aa = torch.zeros(B, dtype=torch.long)

        v = small_model(seq_emb, x_t, t, mut_pos, mut_aa)
        # Velocity should be zero-centered
        assert torch.allclose(v.mean(dim=1), torch.zeros(B, 3), atol=1e-5)

    def test_gradient_flows(self, small_model):
        B, L, D = 2, 8, 16
        seq_emb = torch.randn(B, L, D)
        x_t = torch.randn(B, L, 3)
        t = torch.rand(B)
        mut_pos = torch.zeros(B, dtype=torch.long)
        mut_aa = torch.zeros(B, dtype=torch.long)

        v = small_model(seq_emb, x_t, t, mut_pos, mut_aa)
        loss = (v ** 2).sum()  # squared to ensure gradient through zero-init output
        loss.backward()

        grad_params = [p for p in small_model.parameters() if p.grad is not None]
        assert len(grad_params) > 0, "No parameters received gradients"


class TestFlowMatcher:
    def test_loss_positive(self, small_model):
        fm = ConditionalFlowMatcher()
        B, L, D = 4, 10, 16
        x_1 = torch.randn(B, L, 3)
        x_1 = x_1 - x_1.mean(dim=1, keepdim=True)
        seq_emb = torch.randn(B, L, D)
        mut_pos = torch.zeros(B, dtype=torch.long)
        mut_aa = torch.zeros(B, dtype=torch.long)

        loss = fm.compute_loss(small_model, x_1, seq_emb, mut_pos, mut_aa)
        assert loss.item() > 0

    def test_loss_decreases_with_training(self, small_model):
        """Smoke test: loss should decrease over a few steps on tiny data."""
        fm = ConditionalFlowMatcher()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

        B, L, D = 4, 8, 16
        x_1 = torch.randn(B, L, 3)
        x_1 = x_1 - x_1.mean(dim=1, keepdim=True)
        seq_emb = torch.randn(B, L, D)
        mut_pos = torch.zeros(B, dtype=torch.long)
        mut_aa = torch.zeros(B, dtype=torch.long)

        losses = []
        for _ in range(20):
            loss = fm.compute_loss(small_model, x_1, seq_emb, mut_pos, mut_aa)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should trend downward (first > last)
        assert losses[0] > losses[-1]


class TestODESampler:
    def test_sample_shape(self, small_model):
        small_model.eval()
        sampler = ODESampler(small_model.forward, n_steps=5, method="euler")

        L, D = 10, 16
        seq_emb = torch.randn(1, L, D)
        mut_pos = torch.zeros(1, dtype=torch.long)
        mut_aa = torch.zeros(1, dtype=torch.long)

        samples = sampler.sample(seq_emb, mut_pos, mut_aa, n_samples=3)
        assert samples.shape == (3, L, 3)

    def test_heun_method(self, small_model):
        small_model.eval()
        sampler = ODESampler(small_model.forward, n_steps=3, method="heun")

        L, D = 8, 16
        seq_emb = torch.randn(1, L, D)
        mut_pos = torch.zeros(1, dtype=torch.long)
        mut_aa = torch.zeros(1, dtype=torch.long)

        samples = sampler.sample(seq_emb, mut_pos, mut_aa, n_samples=2)
        assert samples.shape == (2, L, 3)
