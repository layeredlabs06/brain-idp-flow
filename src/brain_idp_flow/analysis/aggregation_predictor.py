"""Correlation analysis: ESM-2 + PED features vs aggregation rate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def run_correlation_analysis(
    esm2_results: list[dict],
    ped_results: list[dict],
    output_dir: str = "results",
) -> dict:
    """Run full correlation analysis and generate figures.

    Returns summary dict with all correlation results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge ESM-2 and PED results by (target, mutation)
    merged = {}
    for r in esm2_results:
        key = (r["target"], r["mutation"])
        merged[key] = {**r}
    for r in ped_results:
        key = (r["target"], r["mutation"])
        if key in merged:
            merged[key].update({k: v for k, v in r.items()
                                if k not in ("target", "mutation", "pos", "agg_rate")})

    data = list(merged.values())
    n = len(data)

    agg_rates = np.array([d["agg_rate"] for d in data])
    log_agg = np.log(agg_rates + 1e-8)

    # Features to test
    feature_names = {
        # ESM-2 features
        "llr_site": "ESM-2 LLR (site)",
        "delta_ppl": "ESM-2 ΔPPL",
        # PED structural features
        "site_rmsf": "PED RMSF",
        "site_contact_freq": "PED Contact Freq",
        "site_long_range_cf": "PED Long-Range CF",
        "local_rg": "PED Local Rg",
        "beta_propensity": "PED β-propensity",
        "exposure_proxy": "PED Exposure",
        # Trajectory velocity features (Direction A)
        "late_velocity_site": "Late-Stage Velocity",
        "late_velocity_global": "Late-Stage Velocity (global)",
        "convergence_time_site": "Convergence Time",
        "convergence_delay_vs_neighbors": "Convergence Delay",
        "velocity_variance_late": "Velocity Variance (late)",
        # Trajectory contact features (Direction B)
        "switching_rate_site": "Contact Switching Rate",
        "switching_rate_long_range": "LR Contact Switching",
        "contact_order_site": "Contact Formation Order",
        "early_contact_fraction_site": "Early Contact Fraction",
    }

    # Compute correlations
    correlations = {}
    for feat, label in feature_names.items():
        vals = [d.get(feat) for d in data]
        if any(v is None for v in vals):
            continue
        vals = np.array(vals)
        rho, pval = spearmanr(vals, log_agg)
        correlations[feat] = {
            "label": label,
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "significant": pval < 0.05,
            "n": n,
        }

    # Print results
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS (n={n} mutations)")
    print(f"{'='*60}")
    print(f"{'Feature':<25} {'ρ':>8} {'p-value':>10} {'Sig?':>6}")
    print(f"{'-'*60}")
    for feat, info in sorted(correlations.items(), key=lambda x: abs(x[1]["spearman_rho"]), reverse=True):
        sig = "***" if info["p_value"] < 0.001 else "**" if info["p_value"] < 0.01 else "*" if info["p_value"] < 0.05 else ""
        print(f"{info['label']:<25} {info['spearman_rho']:>8.3f} {info['p_value']:>10.4f} {sig:>6}")

    # Find best feature
    best_feat = max(correlations.items(), key=lambda x: abs(x[1]["spearman_rho"]))
    print(f"\nBest predictor: {best_feat[1]['label']} (ρ={best_feat[1]['spearman_rho']:.3f})")

    # === FIGURES ===

    # Figure 1: ESM-2 LLR vs aggregation rate
    _plot_scatter(
        data, "llr_site", "ESM-2 Log-Likelihood Ratio",
        agg_rates, output_dir / "esm2_llr_vs_agg.png"
    )

    # Figure 2: Best PED feature vs aggregation rate
    if best_feat[0] != "llr_site":
        _plot_scatter(
            data, best_feat[0], best_feat[1]["label"],
            agg_rates, output_dir / "best_ped_vs_agg.png"
        )

    # Figure 3: Correlation heatmap
    _plot_heatmap(data, feature_names, agg_rates, output_dir / "correlation_heatmap.png")

    # Figure 4: Multi-panel per-protein analysis
    _plot_per_protein(data, correlations, output_dir / "per_protein.png")

    # Composite score
    composite = _compute_composite(data, correlations, agg_rates)

    summary = {
        "n_mutations": n,
        "correlations": correlations,
        "best_feature": best_feat[0],
        "composite": composite,
    }

    return summary


def _plot_scatter(
    data: list[dict], feat: str, label: str,
    agg_rates: np.ndarray, save_path: Path,
) -> None:
    vals = np.array([d.get(feat, 0) for d in data])
    labels = [f"{d['target']}\n{d['mutation']}" for d in data]

    # Color by protein
    protein_colors = {"tau_K18": "#1f77b4", "asyn": "#ff7f0e", "abeta42": "#2ca02c"}
    colors = [protein_colors.get(d["target"], "gray") for d in data]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(vals, agg_rates, s=80, c=colors, edgecolors="black", linewidth=0.5, zorder=5)

    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (vals[i], agg_rates[i]), fontsize=6,
                    textcoords="offset points", xytext=(5, 5))

    rho, pval = spearmanr(vals, np.log(agg_rates + 1e-8))
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel("Relative Aggregation Rate", fontsize=12)
    ax.set_title(f"{label} vs Aggregation Rate\n"
                 f"Spearman ρ={rho:.3f}, p={pval:.4f} (n={len(data)})",
                 fontsize=13)

    # Legend
    for prot, color in protein_colors.items():
        ax.scatter([], [], c=color, label=prot, edgecolors="black", linewidth=0.5)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def _plot_heatmap(
    data: list[dict], feature_names: dict,
    agg_rates: np.ndarray, save_path: Path,
) -> None:
    log_agg = np.log(agg_rates + 1e-8)
    feats = []
    labels = []
    rhos = []

    for feat, label in feature_names.items():
        vals = [d.get(feat) for d in data]
        if any(v is None for v in vals):
            continue
        rho, _ = spearmanr(vals, log_agg)
        feats.append(feat)
        labels.append(label)
        rhos.append(rho)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#d73027" if r < 0 else "#4575b4" for r in rhos]
    bars = ax.barh(range(len(rhos)), rhos, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Spearman ρ vs log(Aggregation Rate)", fontsize=11)
    ax.set_title("Feature Correlation with Aggregation Propensity", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.5)

    for i, rho in enumerate(rhos):
        ax.text(rho + 0.02 * np.sign(rho), i, f"{rho:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def _plot_per_protein(
    data: list[dict], correlations: dict, save_path: Path,
) -> None:
    proteins = sorted(set(d["target"] for d in data))
    feat = "llr_site"

    fig, axes = plt.subplots(1, len(proteins), figsize=(5 * len(proteins), 5))
    if len(proteins) == 1:
        axes = [axes]

    for ax, prot in zip(axes, proteins):
        prot_data = [d for d in data if d["target"] == prot]
        vals = np.array([d[feat] for d in prot_data])
        rates = np.array([d["agg_rate"] for d in prot_data])
        labels = [d["mutation"] for d in prot_data]

        ax.scatter(vals, rates, s=80, edgecolors="black", linewidth=0.5)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (vals[i], rates[i]), fontsize=8,
                        textcoords="offset points", xytext=(5, 5))

        if len(vals) >= 3:
            rho, pval = spearmanr(vals, np.log(rates + 1e-8))
            ax.set_title(f"{prot}\nρ={rho:.3f}, p={pval:.3f} (n={len(vals)})")
        else:
            ax.set_title(f"{prot} (n={len(vals)})")

        ax.set_xlabel("ESM-2 LLR")
        ax.set_ylabel("Agg. Rate")

    fig.suptitle("Per-Protein ESM-2 LLR vs Aggregation", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def _compute_composite(
    data: list[dict], correlations: dict, agg_rates: np.ndarray,
) -> dict:
    """Rank-based composite score from top 3 features."""
    log_agg = np.log(agg_rates + 1e-8)

    # Top features by |rho|
    top = sorted(correlations.items(), key=lambda x: abs(x[1]["spearman_rho"]), reverse=True)[:3]

    n = len(data)
    composite_ranks = np.zeros(n)

    for feat, info in top:
        vals = np.array([d.get(feat, 0) for d in data])
        # Rank, direction-corrected
        ranks = np.argsort(np.argsort(vals)).astype(float)
        if info["spearman_rho"] < 0:
            ranks = n - 1 - ranks
        composite_ranks += ranks

    rho, pval = spearmanr(composite_ranks, log_agg)

    print(f"\nComposite score (top 3 features): ρ={rho:.3f}, p={pval:.4f}")
    return {
        "features_used": [f[0] for f in top],
        "spearman_rho": float(rho),
        "p_value": float(pval),
    }


def leave_one_protein_out_cv(
    data: list[dict],
    feature_names: dict | None = None,
) -> dict:
    """Leave-one-protein-out cross-validation for aggregation prediction.

    Fits rank-based composite on 2 proteins, evaluates on the held-out protein.
    No model retraining — only the correlation/composite scoring is CV'd.

    Args:
        data: list of mutation dicts (must include "target" and "agg_rate" keys)
        feature_names: feature name -> label mapping (uses default if None)

    Returns:
        dict with per-protein and mean results
    """
    if feature_names is None:
        feature_names = {
            "llr_site": "ESM-2 LLR",
            "late_velocity_site": "Late-Stage Velocity",
            "switching_rate_site": "Contact Switching Rate",
        }

    proteins = sorted(set(d["target"] for d in data))
    results = {}

    print(f"\n{'='*60}")
    print(f"LEAVE-ONE-PROTEIN-OUT CV ({len(proteins)} folds)")
    print(f"{'='*60}")

    all_rhos = []

    for held_out in proteins:
        train_data = [d for d in data if d["target"] != held_out]
        test_data = [d for d in data if d["target"] == held_out]

        if len(test_data) < 3:
            print(f"  {held_out}: skipped (n={len(test_data)} < 3)")
            continue

        train_agg = np.log(np.array([d["agg_rate"] for d in train_data]) + 1e-8)
        test_agg = np.log(np.array([d["agg_rate"] for d in test_data]) + 1e-8)

        # Find best features on training set
        train_corrs = {}
        for feat, label in feature_names.items():
            train_vals = [d.get(feat) for d in train_data]
            if any(v is None for v in train_vals):
                continue
            rho, _ = spearmanr(train_vals, train_agg)
            train_corrs[feat] = rho

        if not train_corrs:
            continue

        # Top 3 features from training
        top_feats = sorted(train_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        # Composite score on test set using training-derived ranks/directions
        n_test = len(test_data)
        composite = np.zeros(n_test)

        for feat, train_rho in top_feats:
            test_vals = np.array([d.get(feat, 0) for d in test_data])
            ranks = np.argsort(np.argsort(test_vals)).astype(float)
            if train_rho < 0:
                ranks = n_test - 1 - ranks
            composite += ranks

        rho_test, pval_test = spearmanr(composite, test_agg)
        all_rhos.append(rho_test)

        results[held_out] = {
            "n_test": n_test,
            "spearman_rho": float(rho_test),
            "p_value": float(pval_test),
            "features_used": [f[0] for f in top_feats],
        }

        print(f"  {held_out} (n={n_test}): ρ={rho_test:.3f}, p={pval_test:.3f}")

    mean_rho = float(np.mean(all_rhos)) if all_rhos else 0.0
    print(f"\n  Mean ρ across folds: {mean_rho:.3f}")

    return {
        "per_protein": results,
        "mean_rho": mean_rho,
        "n_folds": len(all_rhos),
    }
