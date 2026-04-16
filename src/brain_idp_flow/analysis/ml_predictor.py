"""ML-based aggregation rate prediction from combined features.

Replaces naive rank-based composite with proper ML:
- Lasso regression (feature selection + sparse weights)
- Random Forest (non-linear interactions)
- 5-fold CV within protein
- Cross-protein transfer test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


FEATURE_KEYS = [
    "llr_site",
    "delta_rg",
    "delta_rg_650m",
    "delta_rg_product",
    "delta_rg_divergence",
    "site_rmsf",
    "site_contact_freq",
    "site_long_range_cf",
    "local_rg",
    "late_velocity_site",
    "velocity_variance_late",
    "switching_rate_site",
    "switching_rate_long_range",
    "contact_order_site",
    "early_contact_fraction_site",
    "contact_formation_delay",
    "convergence_time_site",
]

# Original 14 features (without cross-scale) for backward compatibility
FEATURE_KEYS_SINGLE_SCALE = [
    k for k in FEATURE_KEYS
    if k not in ("delta_rg_650m", "delta_rg_product", "delta_rg_divergence")
]


def add_cross_scale_features(data: list[dict]) -> list[dict]:
    """Compute cross-scale interaction features from 35M and 650M ΔRg.

    Adds delta_rg_product and delta_rg_divergence to each mutation dict.
    Requires both delta_rg (35M) and delta_rg_650m to be present.

    Args:
        data: list of mutation dicts, each containing delta_rg and delta_rg_650m

    Returns:
        New list of dicts with added cross-scale features (immutable pattern).
    """
    enriched = []
    for d in data:
        entry = dict(d)
        drg_35m = d.get("delta_rg")
        drg_650m = d.get("delta_rg_650m")
        if drg_35m is not None and drg_650m is not None:
            entry["delta_rg_product"] = drg_35m * drg_650m
            entry["delta_rg_divergence"] = abs(drg_35m - drg_650m)
        enriched.append(entry)
    return enriched


def _build_feature_matrix(
    data: list[dict],
    feature_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix X and target y from mutation data.

    Args:
        data: list of mutation dicts
        feature_keys: which features to include

    Returns:
        X: (n, p) feature matrix
        y: (n,) log aggregation rates
        used_keys: list of feature keys actually used (no None values)
    """
    if feature_keys is None:
        feature_keys = FEATURE_KEYS

    # Filter to features that exist for all mutations
    used_keys = []
    for key in feature_keys:
        vals = [d.get(key) for d in data]
        if all(v is not None for v in vals):
            arr = np.array(vals, dtype=float)
            if arr.std() > 1e-10:
                used_keys.append(key)

    if not used_keys:
        raise ValueError(
            f"No valid features found. "
            f"Requested: {feature_keys}, available: {list(data[0].keys()) if data else []}"
        )

    X = np.column_stack([
        np.array([d[key] for d in data], dtype=float)
        for key in used_keys
    ])

    y = np.log(np.array([d["agg_rate"] for d in data], dtype=float) + 1e-8)

    return X, y, used_keys


def _zscore_per_protein(
    X: np.ndarray,
    y: np.ndarray,
    proteins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score normalize X and y within each protein."""
    X_norm = X.copy()
    y_norm = y.copy()

    for prot in np.unique(proteins):
        mask = proteins == prot
        for j in range(X.shape[1]):
            col = X[mask, j]
            std = col.std()
            if std > 1e-10:
                X_norm[mask, j] = (col - col.mean()) / std
            else:
                X_norm[mask, j] = 0.0
        y_col = y[mask]
        y_std = y_col.std()
        if y_std > 1e-10:
            y_norm[mask] = (y_col - y_col.mean()) / y_std

    return X_norm, y_norm


def run_lasso_cv(
    data: list[dict],
    feature_keys: list[str] | None = None,
    n_folds: int = 5,
    normalize_per_protein: bool = True,
) -> dict:
    """Lasso regression with cross-validation.

    Args:
        data: list of mutation dicts
        feature_keys: which features to include
        n_folds: number of CV folds
        normalize_per_protein: z-score within each protein first

    Returns:
        dict with coefficients, CV scores, feature importances
    """
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    X, y, used_keys = _build_feature_matrix(data, feature_keys)
    proteins = np.array([d["target"] for d in data])

    if normalize_per_protein:
        X, y = _zscore_per_protein(X, y, proteins)

    # Standard scale (Lasso needs it)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Lasso with built-in CV for alpha selection
    lasso = LassoCV(cv=n_folds, alphas=np.logspace(-4, 1, 50), max_iter=10000)
    lasso.fit(X_scaled, y)

    # Predictions
    y_pred = lasso.predict(X_scaled)
    rho_train, p_train = spearmanr(y_pred, y)

    # Manual K-fold CV for honest evaluation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_rhos = []
    for train_idx, test_idx in kf.split(X_scaled):
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=lasso.alpha_, max_iter=10000)
        model.fit(X_scaled[train_idx], y[train_idx])
        pred = model.predict(X_scaled[test_idx])
        if len(pred) >= 3:
            rho, _ = spearmanr(pred, y[test_idx])
            if not np.isnan(rho):
                cv_rhos.append(rho)

    mean_cv_rho = float(np.mean(cv_rhos)) if cv_rhos else 0.0

    # Feature importances (absolute coefficient)
    coef_importance = sorted(
        zip(used_keys, lasso.coef_),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    print(f"\n{'='*60}")
    print(f"LASSO REGRESSION (n={len(data)}, p={len(used_keys)})")
    print(f"{'='*60}")
    print(f"Best alpha: {lasso.alpha_:.4f}")
    print(f"Train ρ: {rho_train:.3f} (p={p_train:.4f})")
    print(f"CV ρ (mean {n_folds}-fold): {mean_cv_rho:.3f}")
    print(f"\nFeature weights:")
    for key, coef in coef_importance:
        marker = "***" if abs(coef) > 0.1 else ""
        print(f"  {key:<35} {coef:>8.4f} {marker}")

    return {
        "model": "lasso",
        "alpha": float(lasso.alpha_),
        "train_rho": float(rho_train),
        "cv_mean_rho": mean_cv_rho,
        "cv_rhos": cv_rhos,
        "coefficients": {k: float(v) for k, v in coef_importance},
        "features_used": used_keys,
        "n": len(data),
    }


def run_random_forest_cv(
    data: list[dict],
    feature_keys: list[str] | None = None,
    n_folds: int = 5,
    normalize_per_protein: bool = True,
) -> dict:
    """Random Forest regression with cross-validation.

    Captures non-linear interactions between features.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    X, y, used_keys = _build_feature_matrix(data, feature_keys)
    proteins = np.array([d["target"] for d in data])

    if normalize_per_protein:
        X, y = _zscore_per_protein(X, y, proteins)

    # K-fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_rhos = []
    importances_sum = np.zeros(len(used_keys))

    for train_idx, test_idx in kf.split(X):
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
        )
        rf.fit(X[train_idx], y[train_idx])
        pred = rf.predict(X[test_idx])
        if len(pred) >= 3:
            rho, _ = spearmanr(pred, y[test_idx])
            if not np.isnan(rho):
                cv_rhos.append(rho)
        importances_sum += rf.feature_importances_

    # Full model for feature importances
    rf_full = RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42
    )
    rf_full.fit(X, y)
    y_pred = rf_full.predict(X)
    rho_train, p_train = spearmanr(y_pred, y)

    mean_cv_rho = float(np.mean(cv_rhos)) if cv_rhos else 0.0
    avg_importance = importances_sum / max(n_folds, 1)

    feat_imp = sorted(
        zip(used_keys, avg_importance),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\n{'='*60}")
    print(f"RANDOM FOREST (n={len(data)}, p={len(used_keys)})")
    print(f"{'='*60}")
    print(f"Train ρ: {rho_train:.3f} (p={p_train:.4f})")
    print(f"CV ρ (mean {n_folds}-fold): {mean_cv_rho:.3f}")
    print(f"\nFeature importance (Gini):")
    for key, imp in feat_imp:
        bar = "█" * int(imp * 50)
        print(f"  {key:<35} {imp:>6.3f} {bar}")

    return {
        "model": "random_forest",
        "train_rho": float(rho_train),
        "cv_mean_rho": mean_cv_rho,
        "cv_rhos": cv_rhos,
        "feature_importance": {k: float(v) for k, v in feat_imp},
        "features_used": used_keys,
        "n": len(data),
    }


def run_cross_protein_transfer(
    data: list[dict],
    feature_keys: list[str] | None = None,
    normalize_per_protein: bool = True,
) -> dict:
    """Train on one protein, predict others.

    The ultimate generalization test: can patterns learned from
    Aβ42's 798 DMS mutations predict tau/α-syn disease mutations?
    """
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    X, y, used_keys = _build_feature_matrix(data, feature_keys)
    proteins = np.array([d["target"] for d in data])
    unique_proteins = sorted(set(proteins))

    if normalize_per_protein:
        X, y = _zscore_per_protein(X, y, proteins)

    print(f"\n{'='*60}")
    print(f"CROSS-PROTEIN TRANSFER TEST")
    print(f"{'='*60}")

    results = {}

    for train_prot in unique_proteins:
        train_mask = proteins == train_prot
        test_mask = ~train_mask

        if train_mask.sum() < 10 or test_mask.sum() < 5:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        y_train = y[train_mask]
        y_test = y[test_mask]

        # Lasso (simple, less overfitting risk)
        model = Lasso(alpha=0.1, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rho, pval = spearmanr(y_pred, y_test)

        test_proteins = proteins[test_mask]
        per_prot = {}
        for prot in sorted(set(test_proteins)):
            prot_mask = test_proteins == prot
            if prot_mask.sum() >= 3:
                r, p = spearmanr(y_pred[prot_mask], y_test[prot_mask])
                per_prot[prot] = {"rho": float(r), "p": float(p), "n": int(prot_mask.sum())}

        results[train_prot] = {
            "train_n": int(train_mask.sum()),
            "test_n": int(test_mask.sum()),
            "overall_rho": float(rho),
            "overall_p": float(pval),
            "per_protein": per_prot,
        }

        print(f"\nTrain: {train_prot} (n={train_mask.sum()}) → Test: others (n={test_mask.sum()})")
        print(f"  Overall: ρ={rho:.3f}, p={pval:.4f}")
        for prot, info in per_prot.items():
            print(f"  {prot}: ρ={info['rho']:.3f}, p={info['p']:.3f} (n={info['n']})")

    return results


def run_full_ml_pipeline(
    data: list[dict],
    feature_keys: list[str] | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run complete ML analysis pipeline.

    Args:
        data: list of mutation dicts with features
        feature_keys: which features to use
        output_dir: save figures here

    Returns:
        dict with all results
    """
    results = {}

    # 1. Lasso
    results["lasso"] = run_lasso_cv(data, feature_keys)

    # 2. Random Forest
    results["rf"] = run_random_forest_cv(data, feature_keys)

    # 3. Cross-protein transfer
    results["transfer"] = run_cross_protein_transfer(data, feature_keys)

    # Summary
    print(f"\n{'='*60}")
    print(f"ML PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Lasso CV ρ:  {results['lasso']['cv_mean_rho']:.3f}")
    print(f"RF CV ρ:     {results['rf']['cv_mean_rho']:.3f}")

    # Plot comparison if output_dir
    if output_dir:
        _plot_ml_comparison(results, Path(output_dir))

    return results


def _plot_ml_comparison(results: dict, output_dir: Path) -> None:
    """Plot ML model comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: CV rho comparison
    ax = axes[0]
    models = ["lasso", "rf"]
    labels = ["Lasso", "Random Forest"]
    cv_rhos = [results[m]["cv_rhos"] for m in models]
    means = [results[m]["cv_mean_rho"] for m in models]

    positions = range(len(models))
    bp = ax.boxplot(cv_rhos, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#4575b4", "#d73027"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman ρ (CV fold)")
    ax.set_title("Cross-Validation Performance")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    for i, m in enumerate(means):
        ax.plot(i, m, "k*", markersize=12)

    # Plot 2: Feature importance comparison
    ax = axes[1]
    lasso_coef = results["lasso"]["coefficients"]
    rf_imp = results["rf"]["feature_importance"]
    all_feats = list(lasso_coef.keys())[:8]

    y_pos = range(len(all_feats))
    lasso_vals = [abs(lasso_coef.get(f, 0)) for f in all_feats]
    rf_vals = [rf_imp.get(f, 0) for f in all_feats]

    # Normalize for comparison
    lasso_max = max(lasso_vals) if max(lasso_vals) > 0 else 1
    rf_max = max(rf_vals) if max(rf_vals) > 0 else 1

    ax.barh(y_pos, [v / lasso_max for v in lasso_vals],
            height=0.35, align="center", color="#4575b4", alpha=0.7, label="Lasso |coef|")
    ax.barh([y + 0.35 for y in y_pos], [v / rf_max for v in rf_vals],
            height=0.35, align="center", color="#d73027", alpha=0.7, label="RF importance")
    ax.set_yticks([y + 0.175 for y in y_pos])
    ax.set_yticklabels([f[:25] for f in all_feats], fontsize=8)
    ax.set_xlabel("Normalized Importance")
    ax.set_title("Feature Importance Comparison")
    ax.legend(fontsize=9)

    fig.tight_layout()
    save_path = output_dir / "ml_comparison.png"
    fig.savefig(str(save_path), dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def run_lean_composite(
    data: list[dict],
    output_dir: str | None = None,
) -> dict:
    """Focused ΔRg + ESM-2 LLR composite analysis.

    Tests individual features, their combination, and progressive
    feature addition to find the optimal lean model.

    Args:
        data: list of mutation dicts with delta_rg, llr_site, agg_rate/nucleation_score
        output_dir: save figures here

    Returns:
        dict with all comparison results
    """
    from sklearn.linear_model import LinearRegression, LassoCV
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    if not data:
        raise ValueError("data must not be empty")

    # Target: use nucleation_score if available, else log(agg_rate)
    if "nucleation_score" in data[0]:
        y = np.array([d["nucleation_score"] for d in data], dtype=float)
        target_name = "Nucleation Score"
    else:
        y = np.log(np.array([d["agg_rate"] for d in data], dtype=float) + 1e-8)
        target_name = "log(Aggregation Rate)"

    n = len(data)

    print(f"\n{'='*60}")
    print(f"LEAN COMPOSITE: ΔRg + ESM-2 LLR (n={n})")
    print(f"{'='*60}")

    # Feature sets to compare
    has_650m = all(d.get("delta_rg_650m") is not None for d in data)

    feature_sets = {
        "LLR only": ["llr_site"],
        "ΔRg only": ["delta_rg"],
        "LLR + ΔRg": ["llr_site", "delta_rg"],
        "LLR + ΔRg + switching": ["llr_site", "delta_rg", "switching_rate_site", "switching_rate_long_range"],
    }

    if has_650m:
        feature_sets.update({
            "ΔRg 650M only": ["delta_rg_650m"],
            "Cross-scale ΔRg": ["delta_rg", "delta_rg_650m"],
            "Cross-scale + interaction": ["delta_rg", "delta_rg_650m", "delta_rg_product"],
            "Cross-scale full": ["delta_rg", "delta_rg_650m", "delta_rg_product", "delta_rg_divergence"],
            "LLR + cross-scale": ["llr_site", "delta_rg", "delta_rg_650m"],
            "LLR + cross-scale full": ["llr_site", "delta_rg", "delta_rg_650m", "delta_rg_product", "delta_rg_divergence"],
        })

    feature_sets["All available"] = None  # uses all non-None features

    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, feat_keys in feature_sets.items():
        if feat_keys is None:
            # All available
            feat_keys = [k for k in FEATURE_KEYS if all(d.get(k) is not None for d in data)]
            feat_keys = [k for k in feat_keys if np.array([d[k] for d in data]).std() > 1e-10]

        # Check availability
        available = [k for k in feat_keys if all(d.get(k) is not None for d in data)]
        if not available:
            continue

        X = np.column_stack([np.array([d[k] for d in data], dtype=float) for k in available])

        # Spearman for individual (if single feature)
        if len(available) == 1:
            rho_raw, p_raw = spearmanr(X[:, 0], y)
        else:
            rho_raw, p_raw = None, None

        # 5-fold CV with linear regression
        scaler = StandardScaler()
        cv_rhos_lr = []
        cv_rhos_gb = []

        for train_idx, test_idx in kf.split(X):
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])

            # Linear regression
            lr = LinearRegression()
            lr.fit(X_tr, y[train_idx])
            pred_lr = lr.predict(X_te)
            r, _ = spearmanr(pred_lr, y[test_idx])
            if not np.isnan(r):
                cv_rhos_lr.append(r)

            # Gradient boosting (mild)
            gb = GradientBoostingRegressor(
                n_estimators=50, max_depth=2, learning_rate=0.1,
                min_samples_leaf=10, random_state=42,
            )
            gb.fit(X_tr, y[train_idx])
            pred_gb = gb.predict(X_te)
            r, _ = spearmanr(pred_gb, y[test_idx])
            if not np.isnan(r):
                cv_rhos_gb.append(r)

        mean_lr = float(np.mean(cv_rhos_lr)) if cv_rhos_lr else 0.0
        mean_gb = float(np.mean(cv_rhos_gb)) if cv_rhos_gb else 0.0

        results[name] = {
            "features": available,
            "n_features": len(available),
            "spearman_raw": float(rho_raw) if rho_raw is not None else None,
            "p_raw": float(p_raw) if p_raw is not None else None,
            "cv_rho_linear": mean_lr,
            "cv_rho_gb": mean_gb,
        }

        raw_str = f"raw ρ={rho_raw:.3f}" if rho_raw is not None else ""
        print(f"  {name:<30} p={len(available)}  LR CV ρ={mean_lr:.3f}  "
              f"GB CV ρ={mean_gb:.3f}  {raw_str}")

    # Best model
    best = max(results.items(), key=lambda x: x[1]["cv_rho_gb"])
    print(f"\n  Best: {best[0]} (GB CV ρ={best[1]['cv_rho_gb']:.3f})")

    # Plot comparison
    if output_dir:
        _plot_lean_composite(results, y, data, target_name, Path(output_dir))

    return results


def _plot_lean_composite(
    results: dict,
    y: np.ndarray,
    data: list[dict],
    target_name: str,
    output_dir: Path,
) -> None:
    """Plot lean composite comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: CV ρ comparison across feature sets
    ax = axes[0]
    names = list(results.keys())
    lr_rhos = [results[n]["cv_rho_linear"] for n in names]
    gb_rhos = [results[n]["cv_rho_gb"] for n in names]

    x_pos = np.arange(len(names))
    ax.bar(x_pos - 0.2, lr_rhos, 0.35, label="Linear", color="#4575b4", alpha=0.8)
    ax.bar(x_pos + 0.2, gb_rhos, 0.35, label="GBoosting", color="#d73027", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=8)
    ax.set_ylabel("CV Spearman ρ")
    ax.set_title("Feature Set Comparison")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)

    # Plot 2: LLR vs nucleation score
    ax = axes[1]
    llr_vals = np.array([d.get("llr_site", 0) for d in data])
    is_fad = np.array([d.get("is_fad", False) for d in data])
    ax.scatter(llr_vals[~is_fad], y[~is_fad], s=10, alpha=0.4, c="steelblue", label="non-fAD")
    if is_fad.any():
        ax.scatter(llr_vals[is_fad], y[is_fad], s=60, facecolors="none",
                   edgecolors="red", linewidth=2, label="fAD")
    rho, pval = spearmanr(llr_vals, y)
    ax.set_xlabel("ESM-2 LLR")
    ax.set_ylabel(target_name)
    ax.set_title(f"ESM-2 LLR vs {target_name}\nρ={rho:.3f}, p={pval:.2e}")
    ax.legend(fontsize=9)

    # Plot 3: ΔRg vs nucleation score
    ax = axes[2]
    drg_vals = np.array([d.get("delta_rg", 0) for d in data])
    ax.scatter(drg_vals[~is_fad], y[~is_fad], s=10, alpha=0.4, c="coral", label="non-fAD")
    if is_fad.any():
        ax.scatter(drg_vals[is_fad], y[is_fad], s=60, facecolors="none",
                   edgecolors="red", linewidth=2, label="fAD")
    rho, pval = spearmanr(drg_vals, y)
    ax.set_xlabel("ΔRg (Flow Model)")
    ax.set_ylabel(target_name)
    ax.set_title(f"ΔRg vs {target_name}\nρ={rho:.3f}, p={pval:.2e}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    save_path = output_dir / "lean_composite.png"
    fig.savefig(str(save_path), dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)
