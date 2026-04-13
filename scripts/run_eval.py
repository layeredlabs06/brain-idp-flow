"""Evaluate generated ensembles against PED reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brain_idp_flow.targets import load_targets
from brain_idp_flow.data.ped_loader import load_ped_or_fallback
from brain_idp_flow.eval.compare import compare_ensembles, compare_mutation_effect
from brain_idp_flow.eval.plots import (
    plot_rg_comparison,
    plot_contact_maps,
    plot_3d_traces,
    plot_mutation_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ensembles")
    parser.add_argument("--samples-dir", type=str, default="samples/flow")
    parser.add_argument("--targets-config", type=str, default="configs/targets.yaml")
    parser.add_argument("--out-dir", type=str, default="samples/eval")
    args = parser.parse_args()

    targets = load_targets(args.targets_config)
    samples_dir = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for tid, target in targets.items():
        print(f"\n=== {target.name} ===")

        # Load reference
        ref = load_ped_or_fallback(target.ped_id, target.length)

        # Load WT
        wt_path = samples_dir / f"{tid}_WT.npy"
        if not wt_path.exists():
            print(f"  Skipping {tid}: no WT samples found")
            continue
        wt = np.load(wt_path)

        # Compare WT vs reference
        metrics = compare_ensembles(wt, ref, label=f"{tid}_WT")
        print(f"  WT vs PED: {metrics.summary()}")
        results[f"{tid}_WT"] = {
            "rg_mean": metrics.rg_mean,
            "rg_js": metrics.rg_js_vs_ref,
            "e2e_js": metrics.e2e_js_vs_ref,
            "contact_l1": metrics.contact_map_l1,
        }

        # Mutation comparisons
        mut_ensembles = {}
        for mut in target.mutations:
            mut_path = samples_dir / f"{tid}_{mut.id}.npy"
            if mut_path.exists():
                mut_ens = np.load(mut_path)
                mut_ensembles[mut.id] = mut_ens

                effect = compare_mutation_effect(wt, mut_ens, ref)
                print(f"  {mut.id}: ΔRg={effect['delta_rg_mean']:.2f}  "
                      f"JS={effect['rg_js_wt_vs_mut']:.4f}")
                results[f"{tid}_{mut.id}"] = {
                    **effect,
                    "agg_rate_relative": mut.agg_rate_relative,
                }

        # Plots
        plot_rg_comparison(
            {"PED ref": ref, "FM WT": wt},
            title=f"{target.name} — Rg Distribution",
            save_path=out_dir / f"{tid}_rg.png",
        )
        plot_contact_maps(
            {"PED ref": ref, "FM WT": wt},
            title=f"{target.name} — Contact Frequency",
            save_path=out_dir / f"{tid}_contacts.png",
        )
        plot_3d_traces(
            wt, title=f"{target.name} — FM WT Traces",
            save_path=out_dir / f"{tid}_traces.png",
        )
        if mut_ensembles:
            plot_mutation_comparison(
                wt, mut_ensembles, ref,
                target_name=target.name,
                save_path=out_dir / f"{tid}_mutations.png",
            )

    # Save results
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
