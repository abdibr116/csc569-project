"""Final test-set evaluation, comparison table with deltas, and all 6 plots.

Run directly:
    python src/evaluate.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe backend (must be set before pyplot)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (  # noqa: E402
    RANDOM_SEED,
    RESULTS_DIR,
    load_data,
    load_json,
    save_json,
)


COLORS = {"Default RF": "#2196F3", "RandomizedSearchCV": "#FF9800", "GP-Optimized": "#4CAF50"}

plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
})


def _train_predict_time(
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[RandomForestClassifier, np.ndarray, float]:
    clf = RandomForestClassifier(**params, random_state=RANDOM_SEED, n_jobs=-1)
    start = time.time()
    clf.fit(X_train, y_train)
    fit_elapsed = time.time() - start
    y_pred = clf.predict(X_test)
    return clf, y_pred, fit_elapsed


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_fitness_evolution(logbook_df: pd.DataFrame) -> None:
    plt.figure()
    gens = logbook_df["gen"].values
    plt.plot(gens, logbook_df["max"], label="Best Fitness", color="#4CAF50", linewidth=2)
    plt.plot(gens, logbook_df["avg"], label="Average Fitness", color="#2196F3", linewidth=2)
    plt.fill_between(
        gens,
        logbook_df["avg"] - logbook_df["std"],
        logbook_df["avg"] + logbook_df["std"],
        alpha=0.2,
        color="#2196F3",
        label="Avg +/- Std",
    )
    plt.xlabel("Generation")
    plt.ylabel("F1 Score (weighted)")
    plt.title("GP Fitness Evolution Over Generations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(RESULTS_DIR, "fitness_evolution.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_comparison_barplot(table: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1 (weighted)"]
    methods = list(table["Method"])
    n_methods = len(methods)
    x = np.arange(len(metrics))
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, method in enumerate(methods):
        vals = [table.loc[table["Method"] == method, m].values[0] for m in metrics]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=COLORS.get(method, "#888"))
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.001,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    all_vals = [
        table.loc[table["Method"] == m, met].values[0] for m in methods for met in metrics
    ]
    floor = max(0.0, min(all_vals) - 0.05)
    ax.set_ylim(floor, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison Across Tuning Methods")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    out = os.path.join(RESULTS_DIR, "comparison_barplot.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, label_mapping: dict[str, int]
) -> None:
    inv = {v: k for k, v in label_mapping.items()}
    class_names = [inv[i] for i in sorted(inv.keys())]
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix -- GP-Optimized Random Forest")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    out = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


MAX_FEATURES_INDEX = {"sqrt": 0, "log2": 1, 0.3: 2, 0.5: 3, 0.7: 4, 0.9: 5}


def plot_hyperparameter_convergence(snapshots: dict[str, list[dict]]) -> None:
    gens_sorted = sorted(int(k) for k in snapshots.keys())
    if not gens_sorted:
        print("[WARN] No snapshots, skipping hyperparameter_convergence.png")
        return

    def collect(param_name: str, transform=lambda x: x) -> list[list[float]]:
        return [
            [transform(ind[param_name]) for ind in snapshots[str(g)]] for g in gens_sorted
        ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    panels = [
        ("n_estimators", lambda x: x),
        ("max_features (index)", lambda x: MAX_FEATURES_INDEX.get(x, -1)),
        ("max_depth", lambda x: 31 if x is None else x),
        ("min_samples_split", lambda x: x),
        ("min_samples_leaf", lambda x: x),
    ]
    param_keys = ["n_estimators", "max_features", "max_depth", "min_samples_split", "min_samples_leaf"]

    for ax, (title, transform), key in zip(axes_flat[:5], panels, param_keys):
        data = collect(key, transform)
        ax.boxplot(data, labels=[f"gen {g}" for g in gens_sorted])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes_flat[5].axis("off")
    axes_flat[5].text(
        0.05,
        0.85,
        "Box plots show the\ndistribution of decoded\nhyperparameter values\nacross the 50 individuals\nin each snapshot generation.",
        fontsize=11,
        verticalalignment="top",
    )

    plt.suptitle("Hyperparameter Distribution Across Generations", fontsize=14)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "hyperparameter_convergence.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_feature_importance(importances: dict[str, float]) -> None:
    sorted_items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]

    plt.figure(figsize=(10, 8))
    plt.barh(names[::-1], vals[::-1], color="#4CAF50")
    plt.xlabel("Importance")
    plt.title("Feature Importance -- GP-Optimized Random Forest")
    plt.grid(True, axis="x", alpha=0.3)
    out = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_time_comparison(times: dict[str, float]) -> None:
    methods = list(times.keys())
    secs = [times[m] for m in methods]

    plt.figure()
    bars = plt.bar(methods, secs, color=[COLORS.get(m, "#888") for m in methods])
    for b, v in zip(bars, secs):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(secs) * 0.01,
            f"{v:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.ylabel("Wall-clock time (seconds)")
    plt.title("Computational Cost Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    out = os.path.join(RESULTS_DIR, "time_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main() -> None:
    print("=== Step 5: Final Evaluation ===")
    X_train, X_test, y_train, y_test, feature_names, label_mapping = load_data()

    baseline_data = load_json(os.path.join(RESULTS_DIR, "baseline_results.json"))
    default_result = baseline_data[0]
    rs_result = baseline_data[1]
    gp_data = load_json(os.path.join(RESULTS_DIR, "gp_best_params.json"))

    methods_specs = [
        ("Default RF", default_result["params"], default_result),
        ("RandomizedSearchCV", rs_result["best_params"], rs_result),
        ("GP-Optimized", gp_data["best_params"], gp_data),
    ]

    rows: list[dict[str, Any]] = []
    times: dict[str, float] = {}
    gp_y_pred: np.ndarray | None = None
    gp_clf: RandomForestClassifier | None = None
    default_metrics: dict[str, float] | None = None

    for name, params, src in methods_specs:
        print(f"--- Evaluating {name} ---")
        clf, y_pred, _fit_t = _train_predict_time(params, X_train, y_train, X_test)
        m = _metrics(y_test, y_pred)
        method_time = float(src.get("total_time_seconds", src.get("time_seconds", 0.0)))
        cv_mean = float(src.get("f1_mean_across_seeds", src.get("cv_f1_mean", float("nan"))))
        cv_std = float(src.get("f1_std_across_seeds", src.get("cv_f1_std", float("nan"))))
        n_seeds = int(src.get("n_seeds", 1))
        cv_label = f"{cv_mean:.4f} +/- {cv_std:.4f}"
        if n_seeds > 1:
            cv_label += f" ({n_seeds} seeds)"

        rows.append({
            "Method": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 (weighted)": m["f1"],
            "CV F1 Mean +/- Std": cv_label,
            "Time (s)": method_time,
        })
        times[name] = method_time
        if name == "Default RF":
            default_metrics = m
        if name == "GP-Optimized":
            gp_y_pred = y_pred
            gp_clf = clf

    table = pd.DataFrame(rows)

    # Add explicit before/after delta columns (Default RF = before, others = after)
    assert default_metrics is not None
    for metric_label, key in [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 (weighted)", "f1"),
    ]:
        delta_col = f"Δ {metric_label} vs Default"
        table[delta_col] = table[metric_label] - default_metrics[key]
    # Percentage F1 improvement
    table["% F1 Improvement vs Default"] = (
        (table["F1 (weighted)"] - default_metrics["f1"]) / default_metrics["f1"] * 100.0
    )

    out_csv = os.path.join(RESULTS_DIR, "comparison_table.csv")
    table.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(table.to_string(index=False))

    # Wilcoxon signed-rank test on per-fold CV scores
    def _wilcoxon_safe(gp_scores: list, rs_scores: list) -> dict:
        try:
            if np.allclose(gp_scores, rs_scores):
                raise ValueError("identical")
            stat, p_value = wilcoxon(gp_scores, rs_scores)
            return {"statistic": float(stat), "p_value": float(p_value),
                    "significant_at_0.05": bool(p_value < 0.05)}
        except ValueError:
            return {"statistic": None, "p_value": 1.0, "significant_at_0.05": False,
                    "note": "no difference detected"}

    gp_cv_scores = list(gp_data["cv_scores"])
    rs_cv_scores = list(rs_result["cv_scores"])
    is_multi_seed = "all_seeds_summary" in gp_data

    primary = _wilcoxon_safe(gp_cv_scores, rs_cv_scores)
    test_result: dict[str, Any] = {
        "test": "Wilcoxon signed-rank",
        "n_folds": len(gp_cv_scores),
        "gp_scores": gp_cv_scores,
        "rs_scores": rs_cv_scores,
        **primary,
    }

    if is_multi_seed:
        per_seed_tests = []
        per_seed_cv = gp_data.get("per_seed_cv_scores", {})
        for seed_str, scores in per_seed_cv.items():
            res = _wilcoxon_safe(scores, rs_cv_scores)
            per_seed_tests.append({"seed": int(seed_str), **res})
        sig_count = sum(1 for t in per_seed_tests if t["significant_at_0.05"])
        test_result["per_seed_tests"] = per_seed_tests
        test_result["seeds_significant_count"] = sig_count
        test_result["n_seeds"] = gp_data["n_seeds"]
        test_result["f1_mean_across_seeds"] = gp_data["f1_mean_across_seeds"]
        test_result["f1_std_across_seeds"] = gp_data["f1_std_across_seeds"]
        print(f"Multi-seed Wilcoxon: {sig_count}/{len(per_seed_tests)} seeds significant at 0.05")

    print(f"Primary Wilcoxon: stat={primary.get('statistic')}, p={primary.get('p_value'):.4f}")
    save_json(test_result, os.path.join(RESULTS_DIR, "statistical_test.json"))

    # Classification report for GP model
    inv = {v: k for k, v in label_mapping.items()}
    class_names = [inv[i] for i in sorted(inv.keys())]
    report_text = classification_report(y_test, gp_y_pred, target_names=class_names)
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved: {report_path}")
    print(report_text)

    # Feature importances for GP model
    importances = {
        name: float(imp) for name, imp in zip(feature_names, gp_clf.feature_importances_)
    }
    save_json(importances, os.path.join(RESULTS_DIR, "feature_importances.json"))

    # ---- Plots ----
    logbook_df = pd.read_csv(os.path.join(RESULTS_DIR, "gp_evolution.csv"))
    plot_fitness_evolution(logbook_df)

    plot_comparison_barplot(table)
    plot_confusion_matrix(y_test, gp_y_pred, label_mapping)

    snapshots = load_json(os.path.join(RESULTS_DIR, "gp_population_snapshots.json"))
    plot_hyperparameter_convergence(snapshots)

    plot_feature_importance(importances)
    plot_time_comparison(times)

    print("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
