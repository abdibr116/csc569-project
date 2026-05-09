# 10. Future Work

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
