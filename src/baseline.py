"""Train default RandomForest and run RandomizedSearchCV.

Run directly:
    python src/baseline.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (  # noqa: E402
    CV_FOLDS,
    RANDOM_SEED,
    RESULTS_DIR,
    SCORING_METRIC,
    load_data,
    save_json,
)


PARAM_DISTRIBUTIONS = {
    "n_estimators": list(range(50, 501, 10)),
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9],
    "max_depth": [None] + list(range(3, 31)),
    "min_samples_split": list(range(2, 21)),
    "min_samples_leaf": list(range(1, 11)),
}


def run_default_rf(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    print("--- 3A: Default Random Forest ---")
    clf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    start = time.time()
    scores = cross_val_score(
        clf, X_train, y_train, cv=CV_FOLDS, scoring=SCORING_METRIC, n_jobs=-1
    )
    elapsed = time.time() - start
    result = {
        "method": "Default RF",
        "cv_f1_mean": float(scores.mean()),
        "cv_f1_std": float(scores.std()),
        "cv_scores": scores.tolist(),
        "params": {
            "n_estimators": 100,
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "time_seconds": float(elapsed),
    }
    print(
        f"  CV F1 (weighted): {result['cv_f1_mean']:.4f} +/- {result['cv_f1_std']:.4f} "
        f"(elapsed {elapsed:.1f}s)"
    )
    return result


def run_random_search(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    print("--- 3B: RandomizedSearchCV ---")
    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=50,
        cv=CV_FOLDS,
        scoring=SCORING_METRIC,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )
    start = time.time()
    rs.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Best CV F1: {rs.best_score_:.4f}")
    print(f"  Best params: {rs.best_params_}")

    # Per-fold scores at best params for downstream Wilcoxon test
    best_clf = RandomForestClassifier(
        **rs.best_params_, random_state=RANDOM_SEED, n_jobs=-1
    )
    rs_cv_scores = cross_val_score(
        best_clf, X_train, y_train, cv=CV_FOLDS, scoring=SCORING_METRIC, n_jobs=-1
    )

    best_params_clean = {
        k: (int(v) if isinstance(v, (int, np.integer)) else v)
        for k, v in rs.best_params_.items()
    }

    result = {
        "method": "RandomizedSearchCV",
        "cv_f1_mean": float(rs.best_score_),
        "cv_f1_std": float(rs_cv_scores.std()),
        "cv_scores": rs_cv_scores.tolist(),
        "best_params": best_params_clean,
        "time_seconds": float(elapsed),
    }

    cv_results_df = pd.DataFrame(rs.cv_results_)
    cv_results_df.to_csv(
        os.path.join(RESULTS_DIR, "random_search_cv_results.csv"), index=False
    )
    print(
        f"  Saved: {os.path.join(RESULTS_DIR, 'random_search_cv_results.csv')}"
    )
    return result


def main() -> None:
    print("=== Step 3: Baselines ===")
    X_train, _, y_train, _, _, _ = load_data()
    default_result = run_default_rf(X_train, y_train)
    rs_result = run_random_search(X_train, y_train)
    save_json(
        [default_result, rs_result],
        os.path.join(RESULTS_DIR, "baseline_results.json"),
    )


if __name__ == "__main__":
    main()
