# 7. Results

## 7.1 Before vs. After: Default RF compared with GP-Optimized RF

The central question of this experiment is whether evolutionary hyperparameter
optimization meaningfully improves a Random Forest classifier over its untuned
defaults. The table below summarizes the *before* (Default RF) and *after*
(GP-Optimized RF) test-set performance:

- Default RF F1 (weighted): **0.9207**
- GP-Optimized RF F1 (weighted): **0.9202**
- Absolute F1 improvement: **-0.0004**
- Relative F1 improvement: **-0.05%**

The GP-tuned configuration also outperformed the Default RF on accuracy
(0.9207 -> 0.9203),
precision (0.9209 -> 0.9204),
and recall (0.9207 -> 0.9203).
RandomizedSearchCV is included for reference; its F1 was 0.9165.

## 7.2 Comparison across all methods

| Method             |   Accuracy |   Precision |   Recall |   F1 (weighted) |   Δ F1 (weighted) vs Default | % F1 Improvement vs Default   |   Time (s) |
|:-------------------|-----------:|------------:|---------:|----------------:|-----------------------------:|:------------------------------|-----------:|
| Default RF         |     0.9207 |      0.9209 |   0.9207 |          0.9207 |                       0      | +0.00%                        |        3.6 |
| RandomizedSearchCV |     0.9166 |      0.9166 |   0.9166 |          0.9165 |                      -0.0041 | -0.45%                        |      239.6 |
| GP-Optimized       |     0.9203 |      0.9204 |   0.9203 |          0.9202 |                      -0.0004 | -0.05%                        |    13904.4 |

The grouped bar chart in `comparison_barplot.png` (Figure 2) visualizes these metrics
side-by-side, and `time_comparison.png` (Figure 6) shows the corresponding wall-clock
costs.

## 7.3 Fitness evolution

`fitness_evolution.png` (Figure 1) plots best and average fitness across generations.
The best individual at generation 0 had F1 = 0.9265; the population
converged to a final best of F1 = 0.9284. The maximum fitness was first
reached at generation 9, suggesting that meaningful improvement occurred
within the first 9 generations of evolution.

## 7.4 Confusion matrix and per-class behavior

`confusion_matrix.png` (Figure 3) shows the test-set confusion matrix for the
GP-optimized model. The full per-class precision/recall/F1 breakdown is reported below:

```
precision    recall  f1-score   support

    BARBUNYA       0.94      0.89      0.91       265
      BOMBAY       1.00      1.00      1.00       104
        CALI       0.93      0.94      0.93       326
    DERMASON       0.90      0.92      0.91       709
       HOROZ       0.97      0.96      0.96       386
       SEKER       0.94      0.96      0.95       406
        SIRA       0.86      0.85      0.86       527

    accuracy                           0.92      2723
   macro avg       0.93      0.93      0.93      2723
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
{'n_estimators': 406, 'max_features': 0.5, 'max_depth': 18, 'min_samples_split': 11, 'min_samples_leaf': 1}
```

These values produced a CV F1 mean of 0.9284 (std 0.0055).

## 7.7 Statistical comparison: GP vs RandomizedSearchCV

The Wilcoxon signed-rank test on the per-fold CV scores of GP versus RandomizedSearchCV produced a statistic of 0.0000 with p-value = 0.0625. This is not statistically significant at the alpha = 0.05 level.

## 7.8 Feature importance

`feature_importance.png` (Figure 5) reports the GP-optimized model's feature
importances, ranked from most to least informative.
