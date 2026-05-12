# 6. Experiments and Implementation

## 6.1 Dataset

The Dry Bean Dataset from the UCI Machine Learning Repository (Koklu and Ozkan, 2020) was used. It contains shape descriptors of seven varieties of dry beans extracted from segmented images. After preprocessing, the dataset contained 13611 samples with 16 numeric features and 7 target classes.

The data was split into a training set (10888 samples, 80%) and a held-out test set
(2723 samples, 20%) using stratified sampling with `random_state=42`.
Features were standardized with `StandardScaler` fitted on the training split only.

**Class distribution:**

| Class | Full | Train | Test |
|-------|------|-------|------|
| BARBUNYA | 1322 | 1057 | 265 |
| BOMBAY | 522 | 418 | 104 |
| CALI | 1630 | 1304 | 326 |
| DERMASON | 3546 | 2837 | 709 |
| HOROZ | 1928 | 1542 | 386 |
| SEKER | 2027 | 1621 | 406 |
| SIRA | 2636 | 2109 | 527 |

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
   evaluated with 10-fold stratified cross-validation, scoring on `f1_weighted`.

## 6.4 Optimization Method (Genetic Algorithm)

A genetic algorithm was implemented using the DEAP library. Each individual in the
population is encoded as a vector of five floats in [0, 1] which are decoded into the
five RandomForest hyperparameter values. The evolutionary configuration was:

- Population size: 8
- Generations: 3
- Crossover: blend (alpha=0.5), probability 0.7
- Mutation: Gaussian (sigma=0.1, indpb=0.2), probability 0.2
- Selection: tournament (size=3)
- Elitism: top 2 individuals preserved each generation
- Random seed: 42

Each individual's fitness was its mean 10-fold cross-validated weighted F1 score on the
training split, computed using the same CV configuration as the RandomizedSearchCV
baseline.

The GP optimization was repeated with 2 independent random seeds (42, 123)
to assess robustness. Each seed was run with identical GA configuration. The best
seed's parameters were selected for final test-set evaluation, and results are
reported as mean +/- std across all seeds.

## 6.5 Comparison Protocol

All three methods used identical splits (80/20 stratified, seed 42), the same 10-fold
cross-validation, the same `f1_weighted` scoring metric, and the same random seed.
Final test-set evaluation was performed by training each method's best model
configuration on the full training set and predicting on the held-out test set.
