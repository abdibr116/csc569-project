# 8. Discussion

## 8.1 Did evolutionary search help?

GP-optimized RF outperformed the untuned Default RF by +0.0004 in F1
(weighted), or +0.04% in relative terms. Compared to RandomizedSearchCV,
the GP-tuned model differed by +0.0022 (+0.24%). Whether the
gain over RandomizedSearchCV is practically meaningful depends on the deployment
context: for high-stakes or imbalanced classification tasks even a fraction of a
percentage point can matter, while for many applications the gap is within noise.

## 8.2 Statistical significance

The Wilcoxon signed-rank test (statistic=7.0000, p=0.0371) indicates a statistically significant difference between GP and RandomizedSearchCV at alpha = 0.05.

## 8.3 What configuration did GP converge to?

The best individual decoded to:

- `n_estimators` = 99
- `max_features` = 0.5
- `max_depth` = 25
- `min_samples_split` = 9
- `min_samples_leaf` = 1

These values are consistent with the conventional wisdom of Random Forest tuning: a
moderately large ensemble reduces variance, a constrained `max_features` encourages
tree diversity, and bounded `max_depth` together with non-trivial `min_samples_*`
parameters acts as a soft regularizer that limits overfitting on training folds.

## 8.4 Computational cost

The Default RF took 5.3s to evaluate via 5-fold CV.
RandomizedSearchCV (50 candidates) required 416.3s, while the genetic
algorithm consumed 18772.1s. The GP cost is dominated by the
population_size x generations x fold_count product. Whether this cost is
justified depends on whether the resulting model will be reused -- a one-time
investment that yields a permanent +0.04% F1 gain over the untuned
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
