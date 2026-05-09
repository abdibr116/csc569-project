#!/usr/bin/env python3
"""Run the complete RF-GP optimization pipeline.

Usage:
    python run_all.py [--pop-size N] [--generations G] [--refresh-data]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Native shared libraries (openblas, libgfortran, etc.) extracted from Alpine apks
# live alongside the venv. Inject LD_LIBRARY_PATH for child processes if they exist.
_NATIVE_LIB_DIR = os.path.join(PROJECT_ROOT, ".venv", "lib_native")


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    if os.path.isdir(_NATIVE_LIB_DIR):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            _NATIVE_LIB_DIR + (os.pathsep + existing if existing else "")
        )
    return env


CHECKLIST = [
    "data/dry_bean.csv",
    "data/dataset_source.txt",
    "data/README.md",
    "data/X_train.npy",
    "data/X_test.npy",
    "data/y_train.npy",
    "data/y_test.npy",
    "data/feature_names.json",
    "results/label_mapping.json",
    "results/class_distribution.csv",
    "results/baseline_results.json",
    "results/random_search_cv_results.csv",
    "results/gp_best_params.json",
    "results/gp_evolution.csv",
    "results/gp_population_snapshots.json",
    "results/comparison_table.csv",
    "results/statistical_test.json",
    "results/classification_report.txt",
    "results/feature_importances.json",
    "results/fitness_evolution.png",
    "results/comparison_barplot.png",
    "results/confusion_matrix.png",
    "results/hyperparameter_convergence.png",
    "results/feature_importance.png",
    "results/time_comparison.png",
    "paper_sections/06_experiments.md",
    "paper_sections/07_results.md",
    "paper_sections/08_discussion.md",
    "paper_sections/09_challenges.md",
    "paper_sections/10_future_work.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_cmd = [sys.executable, "src/download_data.py"]
    if args.refresh_data:
        download_cmd.append("--refresh-data")

    gp_cmd = [
        sys.executable,
        "src/gp_optimize.py",
        "--pop-size", str(args.pop_size),
        "--generations", str(args.generations),
        "--seed", str(args.seed),
    ]

    steps = [
        ("Download Dataset", download_cmd),
        ("Preprocess", [sys.executable, "src/preprocess.py"]),
        ("Run Baselines", [sys.executable, "src/baseline.py"]),
        ("Run GP Optimization", gp_cmd),
        ("Evaluate & Plot", [sys.executable, "src/evaluate.py"]),
        ("Draft Paper Sections", [sys.executable, "src/draft_paper.py"]),
    ]

    total_start = time.time()
    for name, cmd in steps:
        print(f"\n{'='*60}")
        print(f"  STEP: {name}")
        print(f"  CMD : {' '.join(cmd)}")
        print(f"{'='*60}\n")
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=_child_env())

    total = time.time() - total_start

    print(f"\n{'='*60}")
    print("  EXECUTION CHECKLIST")
    print(f"{'='*60}")
    missing = []
    for path in CHECKLIST:
        full = os.path.join(PROJECT_ROOT, path)
        ok = os.path.exists(full)
        print(f"  {'✓' if ok else '✗'}  {path}")
        if not ok:
            missing.append(path)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE -- Total time: {total:.1f}s")
    if missing:
        print(f"  WARNING: {len(missing)} expected file(s) missing:")
        for p in missing:
            print(f"    - {p}")
        sys.exit(1)
    else:
        print("  All expected outputs present.")
        print("  Results in:        results/")
        print("  Paper drafts in:   paper_sections/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
