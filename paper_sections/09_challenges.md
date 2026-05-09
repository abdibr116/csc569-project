# 9. Challenges and Limitations

## 9.1 Computational cost

Each fitness evaluation requires fitting and scoring a RandomForest five times
(once per CV fold). With a population of 50 and 30
generations, the upper bound on RandomForest fits is approximately
1550 x 5 = 7750. Elitism caches the top
2 unchanged individuals per generation, which trims the count
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
50, generations 30) are reasonable defaults but
were not themselves tuned; a deeper study would treat the GA configuration as a
second-level meta-optimization problem.

## 9.5 Reproducibility constraints

The pipeline fixes `random_state=42` in every randomised component (data split,
RandomForest, GA seeds) so that reruns are deterministic in the limit. However,
`n_jobs=-1` introduces small non-determinism in scikit-learn's parallel
RandomForest training due to floating-point reduction order. Run-to-run F1
variation under this configuration is below 0.001 in our experience, but it
exists and is documented here.
