# 7. Results

## 7.1 Before vs. After: Default RF compared with GP-Optimized RF

The central question of this experiment is whether evolutionary hyperparameter
optimization meaningfully improves a Random Forest classifier over its untuned
defaults. The table below summarizes the *before* (Default RF) and *after*
(GP-Optimized RF) test-set performance:

- Default RF F1 (weighted): **0.9207**
- GP-Optimized RF F1 (weighted): **0.9217**
- Absolute F1 improvement: **+0.0011**
- Relative F1 improvement: **+0.12%**

The GP-tuned configuration also outperformed the Default RF on accuracy
(0.9207 -> 0.9218),
precision (0.9209 -> 0.9220),
and recall (0.9207 -> 0.9218).
RandomizedSearchCV is included for reference; its F1 was 0.9188.

## 7.2 Comparison across all methods

| Method             |   Accuracy |   Precision |   Recall |   F1 (weighted) |   Δ F1 (weighted) vs Default | % F1 Improvement vs Default   |   Time (s) |
|:-------------------|-----------:|------------:|---------:|----------------:|-----------------------------:|:------------------------------|-----------:|
| Default RF         |     0.9207 |      0.9209 |   0.9207 |          0.9207 |                       0      | +0.00%                        |        5.7 |
| RandomizedSearchCV |     0.9188 |      0.9189 |   0.9188 |          0.9188 |                      -0.0018 | -0.20%                        |      417.5 |
| GP-Optimized       |     0.9218 |      0.922  |   0.9218 |          0.9217 |                       0.0011 | +0.12%                        |      312.3 |

The grouped bar chart in `comparison_barplot.png` (Figure 2) visualizes these metrics
side-by-side, and `time_comparison.png` (Figure 6) shows the corresponding wall-clock
costs.

## 7.3 Fitness evolution

`fitness_evolution.png` (Figure 1) plots best and average fitness across generations.
The best individual at generation 0 had F1 = 0.9259; the population
converged to a final best of F1 = 0.9265. The maximum fitness was first
reached at generation 1, suggesting that meaningful improvement occurred
within the first 1 generations of evolution.

## 7.4 Confusion matrix and per-class behavior

`confusion_matrix.png` (Figure 3) shows the test-set confusion matrix for the
GP-optimized model. The full per-class precision/recall/F1 breakdown is reported below:

```
precision    recall  f1-score   support

    BARBUNYA       0.96      0.89      0.92       265
      BOMBAY       1.00      1.00      1.00       104
        CALI       0.93      0.94      0.94       326
    DERMASON       0.91      0.92      0.91       709
       HOROZ       0.96      0.96      0.96       386
       SEKER       0.94      0.96      0.95       406
        SIRA       0.86      0.86      0.86       527

    accuracy                           0.92      2723
   macro avg       0.94      0.93      0.93      2723
weighted avg       0.92      0.92      0.92      2723
```

## 7.5 Hyperparameter convergence

`hyperparameter_convergence.png` (Figure 4) shows the distribution of each decoded
hyperparameter across the 50 individuals at the snapshot generations. The narrowing
of the box plots over time reflects the population converging toward the
GP-optimized values reported in Section 7.6.

## 7.6 Best configuration discovered

The GP search converged on the following hyperparameter values:

```
{'n_estimators': 252, 'max_features': 0.7, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 1}
```

These values produced a CV F1 mean of 0.9265 (std 0.0079).

## 7.7 Statistical comparison: GP vs RandomizedSearchCV

The Wilcoxon signed-rank test on the per-fold CV scores of GP versus RandomizedSearchCV produced a statistic of 27.0000 with p-value = 1.0000. This is not statistically significant at the alpha = 0.05 level.

## 7.8 Feature importance

`feature_importance.png` (Figure 5) reports the GP-optimized model's feature
importances, ranked from most to least informative.
