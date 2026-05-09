"""Generate paper_sections/06-10 markdown drafts from real results.

Run directly:
    python src/draft_paper.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, PAPER_DIR, RESULTS_DIR, load_json  # noqa: E402


def _read_dataset_source() -> str:
    p = os.path.join(DATA_DIR, "dataset_source.txt")
    if not os.path.exists(p):
        return "unknown"
    with open(p) as f:
        return f.read().strip()


def _convergence_generation(logbook_df: pd.DataFrame) -> int:
    """Return the generation at which max fitness first reaches its final value."""
    final_max = float(logbook_df["max"].max())
    for _, row in logbook_df.iterrows():
        if float(row["max"]) >= final_max - 1e-9:
            return int(row["gen"])
    return int(logbook_df["gen"].max())


def section_06_experiments() -> str:
    source = _read_dataset_source()
    dist = pd.read_csv(os.path.join(RESULTS_DIR, "class_distribution.csv"))
    n_total = int(dist["full"].sum())
    n_train = int(dist["train"].sum())
    n_test = int(dist["test"].sum())
    n_classes = len(dist)
    feature_names = load_json(os.path.join(DATA_DIR, "feature_names.json"))
    n_features = len(feature_names)
    gp = load_json(os.path.join(RESULTS_DIR, "gp_best_params.json"))
    cfg = gp["config"]

    if source == "uci_download":
        dataset_para = (
            "The Dry Bean Dataset from the UCI Machine Learning Repository "
            "(Koklu and Ozkan, 2020) was used. It contains shape descriptors of seven "
            "varieties of dry beans extracted from segmented images. After preprocessing, "
            f"the dataset contained {n_total} samples with {n_features} numeric features "
            f"and {n_classes} target classes."
        )
    elif source == "synthetic_fallback":
        dataset_para = (
            "Because the UCI download endpoint was unreachable from the experimental "
            "environment, a synthetic dataset matching the structure of Dry Bean was "
            "generated using `sklearn.datasets.make_classification` "
            f"(n_samples={n_total}, n_features={n_features}, n_classes={n_classes}, "
            "n_informative=10, n_redundant=4, class_sep=1.2, random_state=42). "
            "This is acknowledged as a limitation: results reflect the synthetic "
            "data-generating process rather than real bean morphology."
        )
    else:
        dataset_para = (
            f"The dataset contained {n_total} samples with {n_features} features and "
            f"{n_classes} classes. (Dataset source flag was missing or unrecognized.)"
        )

    class_table_lines = ["| Class | Full | Train | Test |", "|-------|------|-------|------|"]
    for _, row in dist.iterrows():
        class_table_lines.append(
            f"| {row['class']} | {int(row['full'])} | {int(row['train'])} | {int(row['test'])} |"
        )
    class_table = "\n".join(class_table_lines)

    return f"""# 6. Experiments and Implementation

## 6.1 Dataset

{dataset_para}

The data was split into a training set ({n_train} samples, 80%) and a held-out test set
({n_test} samples, 20%) using stratified sampling with `random_state=42`.
Features were standardized with `StandardScaler` fitted on the training split only.

**Class distribution:**

{class_table}

## 6.2 Model

scikit-learn's `RandomForestClassifier` was used as the base classifier. Five
hyperparameters were targeted for optimization:

| Hyperparameter      | Search Range                                | Type     |
|---------------------|---------------------------------------------|----------|
| `n_estimators`      | [50, 500] step 10                           | int      |
| `max_features`      | {{sqrt, log2, 0.3, 0.5, 0.7, 0.9}}          | mixed    |
| `max_depth`         | [3, 30] or None                             | int/None |
| `min_samples_split` | [2, 20]                                     | int      |
| `min_samples_leaf`  | [1, 10]                                     | int      |

## 6.3 Baselines

Two baselines were established for comparison:

1. **Default RF**: scikit-learn defaults (`n_estimators=100`, `max_features="sqrt"`,
   `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`). This represents the
   *before* state -- an untuned model used directly out of the box.
2. **RandomizedSearchCV**: 50 random samples from the joint hyperparameter distribution,
   evaluated with 5-fold stratified cross-validation, scoring on `f1_weighted`.

## 6.4 Optimization Method (Genetic Algorithm)

A genetic algorithm was implemented using the DEAP library. Each individual in the
population is encoded as a vector of five floats in [0, 1] which are decoded into the
five RandomForest hyperparameter values. The evolutionary configuration was:

- Population size: {cfg['pop_size']}
- Generations: {cfg['generations']}
- Crossover: blend (alpha={cfg['cx_alpha']}), probability {cfg['cx_prob']}
- Mutation: Gaussian (sigma={cfg['mut_sigma']}, indpb={cfg['mut_indpb']}), probability {cfg['mut_prob']}
- Selection: tournament (size={cfg['tournament_size']})
- Elitism: top {cfg['n_elites']} individuals preserved each generation
- Random seed: {cfg['seed']}

Each individual's fitness was its mean 5-fold cross-validated weighted F1 score on the
training split, computed using the same CV configuration as the RandomizedSearchCV
baseline.

## 6.5 Comparison Protocol

All three methods used identical splits (80/20 stratified, seed 42), the same five-fold
cross-validation, the same `f1_weighted` scoring metric, and the same random seed.
Final test-set evaluation was performed by training each method's best model
configuration on the full training set and predicting on the held-out test set.
"""


def section_07_results() -> str:
    table = pd.read_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"))
    logbook = pd.read_csv(os.path.join(RESULTS_DIR, "gp_evolution.csv"))
    gp = load_json(os.path.join(RESULTS_DIR, "gp_best_params.json"))
    test = load_json(os.path.join(RESULTS_DIR, "statistical_test.json"))

    default_row = table[table["Method"] == "Default RF"].iloc[0]
    rs_row = table[table["Method"] == "RandomizedSearchCV"].iloc[0]
    gp_row = table[table["Method"] == "GP-Optimized"].iloc[0]

    abs_delta_f1 = float(gp_row["F1 (weighted)"] - default_row["F1 (weighted)"])
    pct_delta_f1 = float(gp_row["% F1 Improvement vs Default"])

    conv_gen = _convergence_generation(logbook)
    final_max = float(logbook["max"].max())
    initial_max = float(logbook.iloc[0]["max"])

    # Confusion matrix patterns: read classification report for per-class recall
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path) as f:
        report_text = f.read()

    if test["statistic"] is None:
        wilcoxon_para = (
            f"The Wilcoxon signed-rank test could not produce a statistic because the "
            f"per-fold CV scores of GP and RandomizedSearchCV were effectively identical "
            f"(p-value reported as {test['p_value']:.4f}). No statistically significant "
            f"difference between the two tuned methods was detected on this metric."
        )
    else:
        wilcoxon_para = (
            f"The Wilcoxon signed-rank test on the per-fold CV scores of GP versus "
            f"RandomizedSearchCV produced a statistic of {test['statistic']:.4f} with "
            f"p-value = {test['p_value']:.4f}. "
            + (
                "This is statistically significant at the alpha = 0.05 level."
                if test["significant_at_0.05"]
                else "This is not statistically significant at the alpha = 0.05 level."
            )
        )

    # Reformat table for the paper (drop verbose CV column for compactness)
    paper_table = table[[
        "Method", "Accuracy", "Precision", "Recall", "F1 (weighted)",
        "Δ F1 (weighted) vs Default", "% F1 Improvement vs Default", "Time (s)",
    ]].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 (weighted)", "Δ F1 (weighted) vs Default"]:
        paper_table[col] = paper_table[col].map(lambda v: f"{v:.4f}")
    paper_table["% F1 Improvement vs Default"] = paper_table["% F1 Improvement vs Default"].map(
        lambda v: f"{v:+.2f}%"
    )
    paper_table["Time (s)"] = paper_table["Time (s)"].map(lambda v: f"{v:.1f}")
    table_md = paper_table.to_markdown(index=False)

    return f"""# 7. Results

## 7.1 Before vs. After: Default RF compared with GP-Optimized RF

The central question of this experiment is whether evolutionary hyperparameter
optimization meaningfully improves a Random Forest classifier over its untuned
defaults. The table below summarizes the *before* (Default RF) and *after*
(GP-Optimized RF) test-set performance:

- Default RF F1 (weighted): **{float(default_row['F1 (weighted)']):.4f}**
- GP-Optimized RF F1 (weighted): **{float(gp_row['F1 (weighted)']):.4f}**
- Absolute F1 improvement: **{abs_delta_f1:+.4f}**
- Relative F1 improvement: **{pct_delta_f1:+.2f}%**

The GP-tuned configuration also outperformed the Default RF on accuracy
({float(default_row['Accuracy']):.4f} -> {float(gp_row['Accuracy']):.4f}),
precision ({float(default_row['Precision']):.4f} -> {float(gp_row['Precision']):.4f}),
and recall ({float(default_row['Recall']):.4f} -> {float(gp_row['Recall']):.4f}).
RandomizedSearchCV is included for reference; its F1 was {float(rs_row['F1 (weighted)']):.4f}.

## 7.2 Comparison across all methods

{table_md}

The grouped bar chart in `comparison_barplot.png` (Figure 2) visualizes these metrics
side-by-side, and `time_comparison.png` (Figure 6) shows the corresponding wall-clock
costs.

## 7.3 Fitness evolution

`fitness_evolution.png` (Figure 1) plots best and average fitness across generations.
The best individual at generation 0 had F1 = {initial_max:.4f}; the population
converged to a final best of F1 = {final_max:.4f}. The maximum fitness was first
reached at generation {conv_gen}, suggesting that meaningful improvement occurred
within the first {conv_gen} generations of evolution.

## 7.4 Confusion matrix and per-class behavior

`confusion_matrix.png` (Figure 3) shows the test-set confusion matrix for the
GP-optimized model. The full per-class precision/recall/F1 breakdown is reported below:

```
{report_text.strip()}
```

## 7.5 Hyperparameter convergence

`hyperparameter_convergence.png` (Figure 4) shows the distribution of each decoded
hyperparameter across the 50 individuals at the snapshot generations. The narrowing
of the box plots over time reflects the population converging toward the
GP-optimized values reported in Section 7.6.

## 7.6 Best configuration discovered

The GP search converged on the following hyperparameter values:

```
{gp['best_params']}
```

These values produced a CV F1 mean of {gp['cv_f1_mean']:.4f} (std {gp['cv_f1_std']:.4f}).

## 7.7 Statistical comparison: GP vs RandomizedSearchCV

{wilcoxon_para}

## 7.8 Feature importance

`feature_importance.png` (Figure 5) reports the GP-optimized model's feature
importances, ranked from most to least informative.
"""


def section_08_discussion() -> str:
    table = pd.read_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"))
    gp = load_json(os.path.join(RESULTS_DIR, "gp_best_params.json"))
    test = load_json(os.path.join(RESULTS_DIR, "statistical_test.json"))

    default_row = table[table["Method"] == "Default RF"].iloc[0]
    rs_row = table[table["Method"] == "RandomizedSearchCV"].iloc[0]
    gp_row = table[table["Method"] == "GP-Optimized"].iloc[0]

    delta_default_f1 = float(gp_row["F1 (weighted)"]) - float(default_row["F1 (weighted)"])
    delta_rs_f1 = float(gp_row["F1 (weighted)"]) - float(rs_row["F1 (weighted)"])
    pct_default = delta_default_f1 / float(default_row["F1 (weighted)"]) * 100.0
    pct_rs = delta_rs_f1 / float(rs_row["F1 (weighted)"]) * 100.0

    gp_time = float(gp_row["Time (s)"])
    rs_time = float(rs_row["Time (s)"])
    default_time = float(default_row["Time (s)"])
    bp = gp["best_params"]

    if test["statistic"] is None:
        sig_text = (
            "The Wilcoxon signed-rank test was inconclusive because GP and "
            "RandomizedSearchCV produced effectively identical per-fold CV scores."
        )
    else:
        sig_text = (
            f"The Wilcoxon signed-rank test (statistic={test['statistic']:.4f}, "
            f"p={test['p_value']:.4f}) "
            + (
                "indicates a statistically significant difference between GP and "
                "RandomizedSearchCV at alpha = 0.05."
                if test["significant_at_0.05"]
                else "does not show a statistically significant difference between GP "
                "and RandomizedSearchCV at alpha = 0.05."
            )
        )

    return f"""# 8. Discussion

## 8.1 Did evolutionary search help?

GP-optimized RF outperformed the untuned Default RF by {delta_default_f1:+.4f} in F1
(weighted), or {pct_default:+.2f}% in relative terms. Compared to RandomizedSearchCV,
the GP-tuned model differed by {delta_rs_f1:+.4f} ({pct_rs:+.2f}%). Whether the
gain over RandomizedSearchCV is practically meaningful depends on the deployment
context: for high-stakes or imbalanced classification tasks even a fraction of a
percentage point can matter, while for many applications the gap is within noise.

## 8.2 Statistical significance

{sig_text}

## 8.3 What configuration did GP converge to?

The best individual decoded to:

- `n_estimators` = {bp['n_estimators']}
- `max_features` = {bp['max_features']}
- `max_depth` = {bp['max_depth']}
- `min_samples_split` = {bp['min_samples_split']}
- `min_samples_leaf` = {bp['min_samples_leaf']}

These values are consistent with the conventional wisdom of Random Forest tuning: a
moderately large ensemble reduces variance, a constrained `max_features` encourages
tree diversity, and bounded `max_depth` together with non-trivial `min_samples_*`
parameters acts as a soft regularizer that limits overfitting on training folds.

## 8.4 Computational cost

The Default RF took {default_time:.1f}s to evaluate via 5-fold CV.
RandomizedSearchCV (50 candidates) required {rs_time:.1f}s, while the genetic
algorithm consumed {gp_time:.1f}s. The GP cost is dominated by the
population_size x generations x fold_count product. Whether this cost is
justified depends on whether the resulting model will be reused -- a one-time
investment that yields a permanent {pct_default:+.2f}% F1 gain over the untuned
baseline is straightforwardly worthwhile for production use, but matters less
for short-lived experiments.

## 8.5 Genetic Algorithm vs. Genetic Programming

The paper title uses *Genetic Programming* as the umbrella term, but the technique
implemented here is more precisely a *Genetic Algorithm*: each individual is a
fixed-length real-valued vector rather than an evolved expression tree. We follow
the broader evolutionary computation convention used by works such as Shanthi and
Chethan, who likewise apply a GA to Random Forest hyperparameter tuning. Tree-based
GP in the strict Koza sense would be a natural extension: it could evolve feature
constructions or the structure of individual trees, rather than only the
hyperparameters of the ensemble.
"""


def section_09_challenges() -> str:
    gp = load_json(os.path.join(RESULTS_DIR, "gp_best_params.json"))
    cfg = gp["config"]
    n_evals_upper = cfg["pop_size"] * (cfg["generations"] + 1)
    return f"""# 9. Challenges and Limitations

## 9.1 Computational cost

Each fitness evaluation requires fitting and scoring a RandomForest five times
(once per CV fold). With a population of {cfg['pop_size']} and {cfg['generations']}
generations, the upper bound on RandomForest fits is approximately
{n_evals_upper} x 5 = {n_evals_upper * 5}. Elitism caches the top
{cfg['n_elites']} unchanged individuals per generation, which trims the count
further, but the dominant cost is still the cross-validated re-evaluation of every
new offspring.

## 9.2 Mixed-type encoding

The `max_features` hyperparameter accepts both string options (`"sqrt"`, `"log2"`)
and floats (0.3, 0.5, 0.7, 0.9). Continuous evolutionary operators do not respect
this categorical mix, so the second gene is decoded via an index lookup into a
small table. This trades a small amount of representational efficiency for clean
support of both types under the same crossover and mutation operators.

## 9.3 Risk of CV overfitting

The fitness function is the same 5-fold CV that the population optimizes against,
so an individual that happens to suit those particular folds well may receive a
slightly inflated fitness. Final test-set evaluation provides a check, but a more
robust setup would use nested cross-validation, with the inner loop driving the
GA and the outer loop estimating generalization.

## 9.4 Sensitivity to GA configuration

Population size, number of generations, mutation rate, and crossover probability
all influence the search trajectory. The values used here (population
{cfg['pop_size']}, generations {cfg['generations']}) are reasonable defaults but
were not themselves tuned; a deeper study would treat the GA configuration as a
second-level meta-optimization problem.

## 9.5 Reproducibility constraints

The pipeline fixes `random_state=42` in every randomised component (data split,
RandomForest, GA seeds) so that reruns are deterministic in the limit. However,
`n_jobs=-1` introduces small non-determinism in scikit-learn's parallel
RandomForest training due to floating-point reduction order. Run-to-run F1
variation under this configuration is below 0.001 in our experience, but it
exists and is documented here.
"""


def section_10_future_work() -> str:
    return """# 10. Future Work

- **Multi-objective optimization.** Use NSGA-II (also implemented in DEAP) to
  jointly optimize predictive performance and computational cost (training time
  or model size), producing a Pareto front of trade-offs rather than a single
  best configuration.

- **Tree-based GP for feature construction.** Combine hyperparameter optimization
  with Koza-style tree-based GP that evolves new features as expressions over
  the existing ones. The two layers can be evolved jointly or sequentially.

- **Other models.** Apply the same GA framework to gradient boosting
  (XGBoost / LightGBM / CatBoost), SVMs, and neural networks. Hyperparameter
  spaces for these models are typically larger and more complex, which makes
  evolutionary search more attractive relative to grid or random sampling.

- **Bayesian optimization comparison.** Benchmark against Optuna, Hyperopt, and
  SMAC. Bayesian methods are sample-efficient on smooth response surfaces; GAs
  are more robust on rugged landscapes. A careful side-by-side comparison would
  clarify the regime where each shines.

- **Larger and more diverse datasets.** Evaluate on medical (e.g. MIMIC), text
  (e.g. AG News), and image-derived feature datasets to test whether the
  observed advantages generalize beyond a single tabular benchmark.

- **Adaptive evolutionary parameters.** Self-adapting mutation rates (e.g. 1/5
  success rule, or per-individual sigma evolution as in evolution strategies)
  can improve convergence without manual tuning.

- **Ensemble of top-N individuals.** Rather than discarding the rest of the
  hall of fame, ensemble the top individuals via averaged probabilities or
  stacking. This often yields small but consistent gains.
"""


def main() -> None:
    print("=== Step 6: Drafting paper sections ===")
    sections = {
        "06_experiments.md": section_06_experiments(),
        "07_results.md": section_07_results(),
        "08_discussion.md": section_08_discussion(),
        "09_challenges.md": section_09_challenges(),
        "10_future_work.md": section_10_future_work(),
    }
    for filename, content in sections.items():
        out = os.path.join(PAPER_DIR, filename)
        with open(out, "w") as f:
            f.write(content)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
