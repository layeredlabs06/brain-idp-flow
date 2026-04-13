"""Download PED ensemble data for brain IDP targets."""

from __future__ import annotations

import argparse
from pathlib import Path

from brain_idp_flow.targets import load_targets
from brain_idp_flow.data.ped_loader import download_ped_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PED ensembles")
    parser.add_argument(
        "--targets", type=str, default="tau_K18,asyn,abeta42",
        help="Comma-separated target IDs"
    )
    parser.add_argument("--config", type=str, default="configs/targets.yaml")
    parser.add_argument("--cache-dir", type=str, default="data/ped")
    args = parser.parse_args()

    targets = load_targets(args.config)
    target_ids = [t.strip() for t in args.targets.split(",")]

    for tid in target_ids:
        if tid not in targets:
            print(f"Warning: {tid} not in config, skipping")
            continue
        target = targets[tid]
        print(f"Fetching PED {target.ped_id} for {target.name}...")
        try:
            path = download_ped_ensemble(target.ped_id, args.cache_dir)
            print(f"  -> {path}")
        except Exception as e:
            print(f"  -> FAILED: {e}")


if __name__ == "__main__":
    main()
