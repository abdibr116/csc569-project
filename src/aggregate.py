"""Aggregate multi-seed GP results into a single results/ output.

Reads results/seed_*/gp_best_params.json, selects the best seed,
and writes aggregated files that evaluate.py and draft_paper.py expect.

Run directly:
    python src/aggregate.py [--seeds 42 123 456 789 1024]
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import RESULTS_DIR, load_json, save_json  # noqa: E402

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def find_seed_dirs(seeds: list[int]) -> list[tuple[int, str]]:
    """Return (seed, directory_path) pairs for seeds that have results."""
    found = []
    for s in seeds:
        d = os.path.join(RESULTS_DIR, f"seed_{s}")
        gp_file = os.path.join(d, "gp_best_params.json")
        if os.path.exists(gp_file):
            found.append((s, d))
        else:
            print(f"[WARN] Missing results for seed {s} at {gp_file}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    print("=== Aggregating multi-seed GP results ===")
    seed_dirs = find_seed_dirs(args.seeds)
    if not seed_dirs:
        raise RuntimeError("No seed results found. Run GP with --seeds first.")

    print(f"Found {len(seed_dirs)} seed(s): {[s for s, _ in seed_dirs]}")

    all_results = []
    for seed, d in seed_dirs:
        gp = load_json(os.path.join(d, "gp_best_params.json"))
        all_results.append((seed, d, gp))

    best_seed, best_dir, best_gp = max(all_results, key=lambda x: x[2]["best_fitness"])
    print(f"Best seed: {best_seed} (CV F1 = {best_gp['best_fitness']:.4f})")

    f1_means = [gp["cv_f1_mean"] for _, _, gp in all_results]
    f1_mean_across = float(sum(f1_means) / len(f1_means))
    f1_std_across = float(pd.Series(f1_means).std())
    total_time = sum(gp["time_seconds"] for _, _, gp in all_results)

    seeds_summary = [
        {
            "seed": seed,
            "best_fitness": gp["best_fitness"],
            "cv_f1_mean": gp["cv_f1_mean"],
            "cv_f1_std": gp["cv_f1_std"],
            "best_params": gp["best_params"],
            "time_seconds": gp["time_seconds"],
        }
        for seed, _, gp in all_results
    ]

    per_seed_cv_scores = {
        seed: gp["cv_scores"] for seed, _, gp in all_results
    }

    aggregated = dict(best_gp)
    aggregated["all_seeds_summary"] = seeds_summary
    aggregated["per_seed_cv_scores"] = per_seed_cv_scores
    aggregated["f1_mean_across_seeds"] = f1_mean_across
    aggregated["f1_std_across_seeds"] = f1_std_across
    aggregated["total_time_seconds"] = total_time
    aggregated["best_seed"] = best_seed
    aggregated["n_seeds"] = len(seed_dirs)

    save_json(aggregated, os.path.join(RESULTS_DIR, "gp_best_params.json"))

    # Copy best seed's evolution + snapshots to top-level results/
    for fname in ("gp_evolution.csv", "gp_population_snapshots.json"):
        src = os.path.join(best_dir, fname)
        dst = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")

    print(f"Aggregation complete: {len(seed_dirs)} seeds, best={best_seed}")
    print(f"  F1 across seeds: {f1_mean_across:.4f} +/- {f1_std_across:.4f}")
    print(f"  Total GP time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
