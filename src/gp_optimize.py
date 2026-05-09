"""DEAP-based genetic algorithm for RandomForest hyperparameter optimization.

Run directly:
    python src/gp_optimize.py [--pop-size N] [--generations G] [--seed S]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (  # noqa: E402
    CV_FOLDS,
    RANDOM_SEED,
    RESULTS_DIR,
    SCORING_METRIC,
    load_data,
    save_json,
)


MAX_FEATURES_OPTIONS: list[Any] = ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]


def decode_individual(individual: list[float]) -> dict[str, Any]:
    """Decode a 5-gene [0,1] vector into RandomForest hyperparameters."""
    g = individual

    n_estimators = int(50 + g[0] * (500 - 50))

    mf_idx = int(g[1] * (len(MAX_FEATURES_OPTIONS) - 1))
    mf_idx = min(mf_idx, len(MAX_FEATURES_OPTIONS) - 1)
    max_features = MAX_FEATURES_OPTIONS[mf_idx]

    if g[2] > 0.95:
        max_depth: int | None = None
    else:
        max_depth = int(3 + g[2] * (30 - 3) / 0.95)

    min_samples_split = int(2 + g[3] * (20 - 2))
    min_samples_leaf = int(1 + g[4] * (10 - 1))

    return {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }


def evaluate_individual(
    individual: list[float],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float]:
    """Fitness = mean 5-fold CV F1-weighted with the decoded RF."""
    params = decode_individual(individual)
    try:
        clf = RandomForestClassifier(**params, random_state=RANDOM_SEED, n_jobs=-1)
        scores = cross_val_score(
            clf, X_train, y_train, cv=CV_FOLDS, scoring=SCORING_METRIC, n_jobs=-1
        )
        return (float(scores.mean()),)
    except Exception as e:
        print(f"  [WARN] Individual failed ({params}): {e}")
        return (0.0,)


def clamp(individual: list[float]) -> list[float]:
    """Clamp gene values to [0, 1] in-place."""
    for i in range(len(individual)):
        individual[i] = max(0.0, min(1.0, individual[i]))
    return individual


def _make_toolbox(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cx_alpha: float,
    mut_sigma: float,
    mut_indpb: float,
    tournament_size: int,
) -> base.Toolbox:
    # creator classes are module-level; recreate idempotently
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=5,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, X_train=X_train, y_train=y_train)
    toolbox.register("mate", tools.cxBlend, alpha=cx_alpha)
    toolbox.register(
        "mutate", tools.mutGaussian, mu=0.0, sigma=mut_sigma, indpb=mut_indpb
    )
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    return toolbox


def run_evolution(
    X_train: np.ndarray,
    y_train: np.ndarray,
    pop_size: int,
    n_generations: int,
    cx_prob: float,
    mut_prob: float,
    n_elites: int,
    cx_alpha: float,
    mut_sigma: float,
    mut_indpb: float,
    tournament_size: int,
    seed: int,
) -> tuple[list, tools.Logbook, tools.HallOfFame, dict[int, list[dict]]]:
    random.seed(seed)
    np.random.seed(seed)

    toolbox = _make_toolbox(
        X_train, y_train, cx_alpha, mut_sigma, mut_indpb, tournament_size
    )

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(n_elites)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]

    # Snapshot generations: 0, 1/3, 2/3, final (deduped, in range)
    snapshot_gens = sorted(
        {0, n_generations // 3, (2 * n_generations) // 3, n_generations}
    )
    snapshots: dict[int, list[dict]] = {}

    # Evaluate initial pop
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    print(logbook.stream)

    if 0 in snapshot_gens:
        snapshots[0] = [decode_individual(ind) for ind in pop]

    for gen in range(1, n_generations + 1):
        offspring = toolbox.select(pop, len(pop) - n_elites)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                clamp(child1)
                clamp(child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                clamp(mutant)
                del mutant.fitness.values

        invalids = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalids))
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit

        elites = list(map(toolbox.clone, hof.items))
        pop[:] = elites + offspring
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalids), **record)
        print(logbook.stream)

        if gen in snapshot_gens:
            snapshots[gen] = [decode_individual(ind) for ind in pop]

    return pop, logbook, hof, snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--cx-prob", type=float, default=0.7)
    parser.add_argument("--mut-prob", type=float, default=0.2)
    parser.add_argument("--n-elites", type=int, default=2)
    parser.add_argument("--cx-alpha", type=float, default=0.5)
    parser.add_argument("--mut-sigma", type=float, default=0.1)
    parser.add_argument("--mut-indpb", type=float, default=0.2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print("=== Step 4: GP (GA) Optimization ===")
    print(
        f"Pop={args.pop_size}, Gen={args.generations}, CxP={args.cx_prob}, "
        f"MutP={args.mut_prob}, Seed={args.seed}"
    )

    X_train, _, y_train, _, _, _ = load_data()

    start = time.time()
    pop, logbook, hof, snapshots = run_evolution(
        X_train,
        y_train,
        pop_size=args.pop_size,
        n_generations=args.generations,
        cx_prob=args.cx_prob,
        mut_prob=args.mut_prob,
        n_elites=args.n_elites,
        cx_alpha=args.cx_alpha,
        mut_sigma=args.mut_sigma,
        mut_indpb=args.mut_indpb,
        tournament_size=args.tournament_size,
        seed=args.seed,
    )
    elapsed = time.time() - start
    print(f"GP elapsed: {elapsed:.1f}s")

    best_ind = hof[0]
    best_params = decode_individual(best_ind)
    best_fitness = float(best_ind.fitness.values[0])
    print(f"Best params: {best_params}")
    print(f"Best fitness (CV F1 mean): {best_fitness:.4f}")

    # Re-run 5-fold CV with best params for per-fold scores
    best_clf = RandomForestClassifier(
        **best_params, random_state=RANDOM_SEED, n_jobs=-1
    )
    cv_scores = cross_val_score(
        best_clf, X_train, y_train, cv=CV_FOLDS, scoring=SCORING_METRIC, n_jobs=-1
    )

    save_json(
        {
            "best_params": best_params,
            "best_fitness": best_fitness,
            "cv_scores": cv_scores.tolist(),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "raw_genes": list(map(float, best_ind)),
            "time_seconds": float(elapsed),
            "config": {
                "pop_size": args.pop_size,
                "generations": args.generations,
                "cx_prob": args.cx_prob,
                "mut_prob": args.mut_prob,
                "n_elites": args.n_elites,
                "cx_alpha": args.cx_alpha,
                "mut_sigma": args.mut_sigma,
                "mut_indpb": args.mut_indpb,
                "tournament_size": args.tournament_size,
                "seed": args.seed,
            },
        },
        os.path.join(RESULTS_DIR, "gp_best_params.json"),
    )

    # Logbook -> CSV
    logbook_df = pd.DataFrame(logbook)
    logbook_df.to_csv(os.path.join(RESULTS_DIR, "gp_evolution.csv"), index=False)
    print(f"Saved: {os.path.join(RESULTS_DIR, 'gp_evolution.csv')}")

    # Snapshots: keys must be JSON strings
    snapshots_serializable = {str(k): v for k, v in snapshots.items()}
    save_json(
        snapshots_serializable,
        os.path.join(RESULTS_DIR, "gp_population_snapshots.json"),
    )


if __name__ == "__main__":
    main()
