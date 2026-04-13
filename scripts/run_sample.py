"""Sample ensembles from trained model for all targets and mutations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from brain_idp_flow.sample import load_model, sample_ensemble
from brain_idp_flow.targets import load_targets
from brain_idp_flow.features.seq_embed import ESM2Embedder
from brain_idp_flow.data.dataset import ProteinEnsembleDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample protein ensembles")
    parser.add_argument("--config", type=str, default="configs/flow.yaml")
    parser.add_argument("--targets-config", type=str, default="configs/targets.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--target", type=str, default=None, help="Single target ID")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", type=str, default="euler")
    parser.add_argument("--out-dir", type=str, default="samples/flow")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    model = load_model(config, args.ckpt, device)
    embedder = ESM2Embedder(device=device)

    targets = load_targets(args.targets_config)
    if args.target:
        targets = {args.target: targets[args.target]}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for tid, target in targets.items():
        print(f"\n=== {target.name} ===")

        # WT ensemble
        seq_emb = embedder.embed_single(target.sequence)
        wt_coords = sample_ensemble(
            model, seq_emb, mut_pos=0, mut_aa=0,
            n_samples=args.n_samples, n_steps=args.steps,
            method=args.method, device=device,
        )
        np.save(out_dir / f"{tid}_WT.npy", wt_coords)
        print(f"  WT: {wt_coords.shape}")

        summary[f"{tid}_WT"] = {"shape": list(wt_coords.shape)}

        # Mutant ensembles
        for mut in target.mutations:
            mut_seq = target.mutant_sequence(mut)
            mut_emb = embedder.embed_single(mut_seq)
            aa_idx = ProteinEnsembleDataset.AA_TO_IDX.get(mut.mt, 0)

            mut_coords = sample_ensemble(
                model, mut_emb, mut_pos=mut.pos, mut_aa=aa_idx,
                n_samples=args.n_samples, n_steps=args.steps,
                method=args.method, device=device,
            )
            np.save(out_dir / f"{tid}_{mut.id}.npy", mut_coords)
            print(f"  {mut.id}: {mut_coords.shape}")

            summary[f"{tid}_{mut.id}"] = {
                "shape": list(mut_coords.shape),
                "agg_rate_relative": mut.agg_rate_relative,
            }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
