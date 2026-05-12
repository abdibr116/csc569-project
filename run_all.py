#!/usr/bin/env python3
"""Run the complete RF-GP optimization pipeline.

Usage:
    python run_all.py [--pop-size N] [--generations G] [--refresh-data]
"""
from __future__ import annotations

import argparse
import datetime
import json
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


def _write_progress(path: str | None, record: dict) -> None:
    if not path:
        return
    record.setdefault("timestamp", datetime.datetime.utcnow().isoformat() + "Z")
    with open(os.path.join(PROJECT_ROOT, path), "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def _run_step(name: str, cmd: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=_child_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Run GP with multiple seeds and aggregate (e.g., --seeds 42 123 456 789 1024)",
    )
    parser.add_argument("--progress-file", type=str, default=None,
                        help="Path to JSONL progress file (relative to project root)")
    args = parser.parse_args()

    pf = args.progress_file
    seeds = args.seeds or [args.seed]

    _write_progress(pf, {
        "type": "experiment", "status": "started",
        "config": {"pop_size": args.pop_size, "generations": args.generations, "seeds": seeds},
    })

    download_cmd = [sys.executable, "src/download_data.py"]
    if args.refresh_data:
        download_cmd.append("--refresh-data")

    total_start = time.time()

    try:
        # Steps 1-3: always run once (deterministic)
        for step_name, cmd in [
            ("download", download_cmd),
            ("preprocess", [sys.executable, "src/preprocess.py"]),
            ("baseline", [sys.executable, "src/baseline.py"]),
        ]:
            _write_progress(pf, {"type": "step", "step": step_name, "status": "running"})
            _run_step(step_name.title(), cmd)
            _write_progress(pf, {"type": "step", "step": step_name, "status": "completed"})

        # Step 4: GP — single-seed or multi-seed
        if args.seeds:
            for seed in args.seeds:
                gp_cmd = [
                    sys.executable, "src/gp_optimize.py",
                    "--pop-size", str(args.pop_size),
                    "--generations", str(args.generations),
                    "--seed", str(seed),
                    "--results-dir", os.path.join("results", f"seed_{seed}"),
                ]
                if pf:
                    gp_cmd += ["--progress-file", pf]
                _write_progress(pf, {"type": "step", "step": "gp", "status": "running", "seed": seed})
                _run_step(f"GP Optimization (seed={seed})", gp_cmd)
                _write_progress(pf, {"type": "step", "step": "gp", "status": "completed", "seed": seed})
            _write_progress(pf, {"type": "step", "step": "aggregate", "status": "running"})
            agg_cmd = [sys.executable, "src/aggregate.py",
                        "--seeds"] + [str(s) for s in args.seeds]
            _run_step("Aggregate multi-seed results", agg_cmd)
            _write_progress(pf, {"type": "step", "step": "aggregate", "status": "completed"})
        else:
            gp_cmd = [
                sys.executable, "src/gp_optimize.py",
                "--pop-size", str(args.pop_size),
                "--generations", str(args.generations),
                "--seed", str(args.seed),
            ]
            if pf:
                gp_cmd += ["--progress-file", pf]
            _write_progress(pf, {"type": "step", "step": "gp", "status": "running", "seed": args.seed})
            _run_step("GP Optimization", gp_cmd)
            _write_progress(pf, {"type": "step", "step": "gp", "status": "completed", "seed": args.seed})

        # Steps 5-6: evaluate and draft
        for step_name, cmd in [
            ("evaluate", [sys.executable, "src/evaluate.py"]),
            ("draft_paper", [sys.executable, "src/draft_paper.py"]),
        ]:
            _write_progress(pf, {"type": "step", "step": step_name, "status": "running"})
            _run_step(step_name.replace("_", " ").title(), cmd)
            _write_progress(pf, {"type": "step", "step": step_name, "status": "completed"})

        total = time.time() - total_start
        _write_progress(pf, {"type": "experiment", "status": "completed", "total_time_s": total})

    except Exception as e:
        total = time.time() - total_start
        _write_progress(pf, {"type": "experiment", "status": "failed", "error": str(e), "total_time_s": total})
        raise

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
