"""Build NPZ dataset from PDB + PED sources with mutation annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from brain_idp_flow.targets import load_targets
from brain_idp_flow.data.ped_loader import load_ped_or_fallback
from brain_idp_flow.data.dataset import ProteinEnsembleDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--out", type=str, default="data/train.npz")
    parser.add_argument("--config", type=str, default="configs/targets.yaml")
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--n", type=int, default=None, help="Limit total samples (debug)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    targets = load_targets(args.config)

    all_coords = []
    all_seq_ids = []
    all_mut_pos = []
    all_mut_aa = []
    seq_id_map = {}

    for tid, target in targets.items():
        print(f"Loading {target.name} (PED: {target.ped_id})...")

        # Load WT ensemble
        ensemble = load_ped_or_fallback(
            target.ped_id, target.length, n_fallback=50
        )

        # Pad/trim to max_len
        K, L, _ = ensemble.shape
        if L > args.max_len:
            ensemble = ensemble[:, :args.max_len, :]
        elif L < args.max_len:
            pad = np.zeros((K, args.max_len - L, 3), dtype=np.float32)
            ensemble = np.concatenate([ensemble, pad], axis=1)

        # Assign seq_id
        sid = len(seq_id_map)
        seq_id_map[sid] = {"target": tid, "mutation": "WT"}

        # Add WT samples
        for frame in ensemble:
            frame_centered = frame - frame.mean(axis=0)
            all_coords.append(frame_centered)
            all_seq_ids.append(sid)
            all_mut_pos.append(0)  # 0 = WT
            all_mut_aa.append(0)   # 0 = WT

        # Add mutant samples (same coordinates, different mutation conditioning)
        # In real training, these would come from MD/PED mutant ensembles
        # For now, we use WT coords with mutation labels as a starting point
        for mut in target.mutations:
            mut_sid = len(seq_id_map)
            seq_id_map[mut_sid] = {"target": tid, "mutation": mut.id}
            aa_idx = ProteinEnsembleDataset.AA_TO_IDX.get(mut.mt, 0)

            for frame in ensemble:
                frame_centered = frame - frame.mean(axis=0)
                all_coords.append(frame_centered)
                all_seq_ids.append(mut_sid)
                all_mut_pos.append(mut.pos)
                all_mut_aa.append(aa_idx)

        print(f"  {tid}: {K} frames, {len(target.mutations)} mutations")

    coords = np.stack(all_coords).astype(np.float32)
    seq_ids = np.array(all_seq_ids, dtype=np.int64)
    mut_pos = np.array(all_mut_pos, dtype=np.int64)
    mut_aa = np.array(all_mut_aa, dtype=np.int64)

    if args.n and len(coords) > args.n:
        idx = np.random.default_rng(42).choice(len(coords), args.n, replace=False)
        coords = coords[idx]
        seq_ids = seq_ids[idx]
        mut_pos = mut_pos[idx]
        mut_aa = mut_aa[idx]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        coords=coords,
        seq_ids=seq_ids,
        mut_pos=mut_pos,
        mut_aa=mut_aa,
    )
    print(f"\nSaved {len(coords)} samples to {args.out}")
    print(f"Shape: coords={coords.shape}, seq_ids={seq_ids.shape}")
    if args.debug:
        print(f"Seq ID map: {seq_id_map}")


if __name__ == "__main__":
    main()
