"""Run flow matching training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from brain_idp_flow.train import train
from brain_idp_flow.targets import load_targets
from brain_idp_flow.features.seq_embed import ESM2Embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train flow matching model")
    parser.add_argument("--config", type=str, default="configs/flow.yaml")
    parser.add_argument("--targets-config", type=str, default="configs/targets.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Pre-compute sequence embeddings
    print("Computing sequence embeddings...")
    targets = load_targets(args.targets_config)
    embedder = ESM2Embedder(device=device)

    seq_embeddings = {}
    sid = 0
    for tid, target in targets.items():
        # WT embedding
        emb = embedder.embed_single(target.sequence)
        seq_embeddings[sid] = emb.cpu()
        sid += 1

        # Mutant embeddings
        for mut in target.mutations:
            mut_seq = target.mutant_sequence(mut)
            emb = embedder.embed_single(mut_seq)
            seq_embeddings[sid] = emb.cpu()
            sid += 1

    print(f"Computed {len(seq_embeddings)} sequence embeddings")

    # Train
    best_ckpt = train(config, seq_embeddings, device, max_steps=args.max_steps)
    print(f"\nBest checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
