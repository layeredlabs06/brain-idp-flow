"""Training loop for mutation-conditioned flow matching."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch import Tensor

from brain_idp_flow.model.structure_head import MutationConditionedStructureHead
from brain_idp_flow.model.flow_matcher import ConditionalFlowMatcher
from brain_idp_flow.data.dataset import ProteinEnsembleDataset


def create_model(config: dict) -> MutationConditionedStructureHead:
    mc = config["model"]
    return MutationConditionedStructureHead(
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        d_seq_in=mc["d_seq_in"],
        dropout=mc["dropout"],
        rbf_bins=mc["rbf_bins"],
        rbf_max=mc["rbf_max"],
        mutation_embed_dim=mc["mutation_embed_dim"],
    )


def train(
    config: dict,
    seq_embeddings: dict[int, Tensor],
    device: torch.device,
    max_steps: Optional[int] = None,
) -> Path:
    """Run training loop.

    Args:
        config: full config dict (model + training + data sections)
        seq_embeddings: {seq_id: (L, D)} pre-computed frozen embeddings
        device: training device
        max_steps: override max steps (for smoke testing)

    Returns:
        Path to best checkpoint.
    """
    tc = config["training"]
    dc = config["data"]
    steps = max_steps or tc["max_steps"]

    # Dataset
    dataset = ProteinEnsembleDataset(
        dc["train_npz"],
        max_len=dc["max_len"],
        augment_rotation=dc.get("augment_rotation", True),
    )
    n_val = max(1, int(len(dataset) * dc.get("val_fraction", 0.1)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_ds.dataset.train()

    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"])

    # Model
    model = create_model(config).to(device)
    fm = ConditionalFlowMatcher()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"]
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=tc["lr"] * 0.01
    )

    # EMA
    ema_model = deepcopy(model)
    ema_decay = tc.get("ema_decay", 0.999)

    # AMP
    scaler = torch.amp.GradScaler("cuda", enabled=tc.get("amp", True) and device.type == "cuda")

    # Logging
    log_dir = Path(tc["log_dir"]) / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_ckpt_path = log_dir / "ckpt_best.pt"
    global_step = 0
    accum_loss = 0.0

    train_iter = iter(train_loader)

    for step in range(steps):
        model.train()

        # Get batch (cycle through data)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        coords = batch["coords"].to(device)
        seq_id = batch["seq_id"]
        mut_pos = batch["mut_pos"].to(device)
        mut_aa = batch["mut_aa"].to(device)

        # Look up pre-computed embeddings
        B, L, _ = coords.shape
        seq_emb = torch.stack([
            _pad_or_trim(seq_embeddings[sid.item()], L).to(device)
            for sid in seq_id
        ])

        with torch.amp.autocast("cuda", enabled=tc.get("amp", True) and device.type == "cuda"):
            loss = fm.compute_loss(model, coords, seq_emb, mut_pos, mut_aa)
            loss = loss / tc["grad_accum"]

        scaler.scale(loss).backward()
        accum_loss += loss.item()

        if (step + 1) % tc["grad_accum"] == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # EMA update
            with torch.no_grad():
                for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                    p_ema.lerp_(p_model, 1 - ema_decay)

            global_step += 1

            if global_step % 50 == 0:
                avg_loss = accum_loss / 50
                lr = optimizer.param_groups[0]["lr"]
                print(f"step {global_step}/{steps // tc['grad_accum']}  "
                      f"loss={avg_loss:.4f}  lr={lr:.2e}")
                accum_loss = 0.0

        # Validation
        if (step + 1) % (tc["val_every"] * tc["grad_accum"]) == 0:
            val_loss = _validate(ema_model, fm, val_loader, seq_embeddings, device)
            print(f"  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_ckpt(ema_model, optimizer, step, best_ckpt_path)
                print(f"  -> saved best checkpoint")

        # Periodic save
        if (step + 1) % (tc["ckpt_every"] * tc["grad_accum"]) == 0:
            _save_ckpt(ema_model, optimizer, step, log_dir / "ckpt_last.pt")

    # Final save
    _save_ckpt(ema_model, optimizer, steps, log_dir / "ckpt_last.pt")
    return best_ckpt_path


def _pad_or_trim(emb: Tensor, target_len: int) -> Tensor:
    L = emb.shape[0]
    if L >= target_len:
        return emb[:target_len]
    pad = torch.zeros(target_len - L, emb.shape[1], dtype=emb.dtype)
    return torch.cat([emb, pad], dim=0)


@torch.no_grad()
def _validate(
    model: nn.Module,
    fm: ConditionalFlowMatcher,
    loader: DataLoader,
    seq_embeddings: dict[int, Tensor],
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        coords = batch["coords"].to(device)
        B, L, _ = coords.shape
        seq_emb = torch.stack([
            _pad_or_trim(seq_embeddings[sid.item()], L).to(device)
            for sid in batch["seq_id"]
        ])
        mut_pos = batch["mut_pos"].to(device)
        mut_aa = batch["mut_aa"].to(device)

        loss = fm.compute_loss(model, coords, seq_emb, mut_pos, mut_aa)
        total_loss += loss.item() * B
        n += B
    return total_loss / max(n, 1)


def _save_ckpt(model: nn.Module, optimizer, step: int, path: Path) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }, str(path))
