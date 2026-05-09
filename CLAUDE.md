# CLAUDE.md — RF Hyperparameter Optimization via Genetic Programming

## Context

This is an academic project for CSC 569 (Selected Topics in AI) at King Saud University, Winter 2026. The paper title is "Optimizing Random Forest Hyperparameters using Genetic Programming for Improved Classification Performance." Student: Rana Almuaied (ID: 447200350).

The paper already has sections 1–5 written (Introduction, Problem Statement, Objectives, Methodology, Literature Review). Sections 6–10 (Experiments/Implementation, Results, Discussion, Challenges, Future Work) are **empty stubs** and need to be filled based on the implementation you build here.

The final deliverable is a 12-page single-column research paper with references (IEEE/ACM style) plus working code.

---

## Project Goal

Build a complete, reproducible Python experiment that:
1. Loads and preprocesses a classification dataset.
2. Trains a baseline Random Forest with default parameters.
3. Tunes RF hyperparameters with RandomizedSearchCV.
4. Tunes RF hyperparameters with a Genetic Algorithm (DEAP library).
5. Compares all three approaches on accuracy, precision, recall, F1, and wall-clock time.
6. Generates publication-ready plots and tables.
7. Drafts the empty paper sections (6–10) as markdown files based on actual results.

---

## Directory Layout

```
rf-gp-optimization/
├── CLAUDE.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── dry_bean.csv           # Downloaded or generated dataset
├── src/
│   ├── download_data.py       # Downloads dataset (with fallback)
│   ├── preprocess.py          # Clean, encode, split
│   ├── baseline.py            # Default RF + RandomizedSearchCV
│   ├── gp_optimize.py         # DEAP-based genetic algorithm optimization
│   ├── evaluate.py            # Final evaluation, comparison table, plots
│   └── utils.py               # Shared constants, helpers, I/O
├── results/                   # All outputs land here (CSV, JSON, PNG)
│   └── .gitkeep
├── paper_sections/            # Drafted markdown for empty paper sections
│   └── .gitkeep
└── run_all.py                 # Single entry point: runs entire pipeline
```

---

## Step 0: Setup

### requirements.txt content:
```
scikit-learn>=1.3.0
deap>=1.4.1
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
requests>=2.28.0
openpyxl>=3.1.0
```

### Install command:
```bash
pip install -r requirements.txt
```

---

## Step 1: Dataset Acquisition (`src/download_data.py`)

### Primary option — Dry Bean Dataset (UCI via direct URL)

The Dry Bean dataset is a multi-class classification problem (7 bean types, 13611 samples, 16 numeric features). It is well-suited for this project because it has enough rows to show meaningful differences between tuning methods, all features are numeric (no complex encoding needed), and it is multi-class which makes F1-weighted more interesting than binary accuracy.

**Download URL:**
```
https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip
```

The ZIP contains an Excel file `Dry_Bean_Dataset.xlsx` with a single sheet. The target column is `Class` with 7 values: SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON.

**Implementation:**
```python
import requests, zipfile, io, os
import pandas as pd

def download_dry_bean(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "dry_bean.csv")
    if os.path.exists(csv_path):
        print(f"Dataset already exists at {csv_path}")
        return csv_path

    url = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
    print(f"Downloading from {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Find the xlsx file inside the zip
        xlsx_files = [f for f in zf.namelist() if f.endswith(".xlsx")]
        if not xlsx_files:
            raise FileNotFoundError("No .xlsx found in ZIP")
        with zf.open(xlsx_files[0]) as f:
            df = pd.read_excel(f, engine="openpyxl")

    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols to {csv_path}")
    return csv_path
```

### Fallback — if download fails (network restrictions, URL changes)

Generate a synthetic dataset using sklearn that mimics the Dry Bean structure. **This fallback MUST be implemented** because the download may fail in restricted environments:

```python
from sklearn.datasets import make_classification
import numpy as np

def generate_fallback_dataset(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "dry_bean.csv")
    X, y = make_classification(
        n_samples=13000,
        n_features=16,
        n_informative=10,
        n_redundant=4,
        n_classes=7,
        n_clusters_per_class=1,
        random_state=42,
        class_sep=1.2,
    )
    feature_names = [
        "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
        "AspectRatio", "Eccentricity", "ConvexArea", "EquivDiameter",
        "Extent", "Solidity", "Roundness", "Compactness",
        "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
    ]
    class_names = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA", "DERMASON"]
    df = pd.DataFrame(X, columns=feature_names)
    df["Class"] = [class_names[label] for label in y]
    df.to_csv(csv_path, index=False)
    print(f"[FALLBACK] Generated synthetic dataset: {len(df)} rows x {len(df.columns)} cols")
    return csv_path
```

**The script must try the real download first, catch any exception, print the error, and fall back to synthetic.** Also save a flag file `data/dataset_source.txt` with either `"uci_download"` or `"synthetic_fallback"` so the paper sections know which to reference.

### data/README.md content to create:
```markdown
# Dataset: Dry Bean Dataset

- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Citation:** Koklu, M. and Ozkan, I.A., 2020. Multiclass classification of dry beans using computer vision and machine learning techniques. Computers and Electronics in Agriculture, 174, 105507.
- **Samples:** 13,611
- **Features:** 16 (all numeric: geometric shape descriptors)
- **Target:** Class (7 bean types: SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON)
- **Task:** Multi-class classification
- **License:** CC BY 4.0
```

---

## Step 2: Preprocessing (`src/preprocess.py`)

### Input
`data/dry_bean.csv`

### Operations
1. Load CSV with `pd.read_csv()`.
2. Print shape, dtypes, and first 5 rows.
3. Check for missing values: `df.isnull().sum()`. Dry Bean has zero missing values. If any column has <5% missing, drop those rows. If >=5%, impute with median. Print what was done.
4. Identify target column: `"Class"`. All other columns are features.
5. Encode target with `sklearn.preprocessing.LabelEncoder`. Save the label mapping to `results/label_mapping.json` as `{"BARBUNYA": 0, "BOMBAY": 1, ...}` — sorted by encoded value.
6. Features are all numeric in Dry Bean — no categorical encoding needed. If a column is non-numeric (dtype object) and not the target, apply `LabelEncoder` to it and print a warning.
7. Scale features with `StandardScaler`. **Fit on train split only** (to avoid data leakage), transform both train and test.
8. Split: `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
9. Save to disk:
   - `results/class_distribution.csv` — value counts of target in full, train, and test sets.
   - `data/X_train.npy`, `data/X_test.npy`, `data/y_train.npy`, `data/y_test.npy` — numpy arrays.
   - `data/feature_names.json` — list of feature column names.
10. Print train/test shapes and class distribution.

### Output files
```
data/X_train.npy, data/X_test.npy, data/y_train.npy, data/y_test.npy
data/feature_names.json
results/class_distribution.csv
results/label_mapping.json
```

---

## Step 3: Baselines (`src/baseline.py`)

### Input
Load `data/X_train.npy`, `data/y_train.npy` using `utils.load_data()`.

### 3A: Default Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time

clf_default = RandomForestClassifier(random_state=42, n_jobs=-1)
start = time.time()
scores = cross_val_score(clf_default, X_train, y_train, cv=5, scoring="f1_weighted")
elapsed = time.time() - start

default_result = {
    "method": "Default RF",
    "cv_f1_mean": float(scores.mean()),
    "cv_f1_std": float(scores.std()),
    "cv_scores": scores.tolist(),
    "params": {
        "n_estimators": 100,
        "max_features": "sqrt",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "time_seconds": elapsed,
}
```

### 3B: RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_distributions = {
    "n_estimators": list(range(50, 501, 10)),        # 50, 60, 70, ..., 500
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9],
    "max_depth": [None] + list(range(3, 31)),         # None, 3, 4, ..., 30
    "min_samples_split": list(range(2, 21)),           # 2, 3, ..., 20
    "min_samples_leaf": list(range(1, 11)),             # 1, 2, ..., 10
}

rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring="f1_weighted",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
start = time.time()
rs.fit(X_train, y_train)
elapsed = time.time() - start

# Also get per-fold scores for the best params (needed for statistical test later)
best_clf = RandomForestClassifier(**rs.best_params_, random_state=42, n_jobs=-1)
rs_cv_scores = cross_val_score(best_clf, X_train, y_train, cv=5, scoring="f1_weighted")

random_search_result = {
    "method": "RandomizedSearchCV",
    "cv_f1_mean": float(rs.best_score_),
    "cv_f1_std": float(rs_cv_scores.std()),
    "cv_scores": rs_cv_scores.tolist(),
    "best_params": {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in rs.best_params_.items()},
    "time_seconds": elapsed,
}
```

### Output
Save both results to `results/baseline_results.json` as a list of two dicts. Also save `results/random_search_cv_results.csv` from `pd.DataFrame(rs.cv_results_)` for potential further analysis.

---

## Step 4: GP Optimization (`src/gp_optimize.py`)

This is the core of the project. Use the DEAP library.

### Important design note

The paper says "Genetic Programming" but the task is optimizing a fixed-length vector of 5 hyperparameters — this is a Genetic Algorithm (GA), not tree-based GP. DEAP supports both. Implement it as a GA using `deap.tools` and `deap.algorithms`. This is standard practice — Shanthi & Chethan [6] in the paper's references also use a GA for hyperparameter tuning. The paper can explain this distinction in the Discussion section.

### Individual representation

Each individual is a list of 5 floats in [0.0, 1.0]. These are decoded into actual hyperparameter values:

```python
def decode_individual(individual):
    """Map [0,1] genes to actual hyperparameter values."""
    g = individual

    # Gene 0 -> n_estimators: [50, 500], integer
    n_estimators = int(50 + g[0] * (500 - 50))

    # Gene 1 -> max_features: map to discrete choices
    max_features_options = ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]
    max_features_idx = int(g[1] * (len(max_features_options) - 1))
    max_features_idx = min(max_features_idx, len(max_features_options) - 1)
    max_features = max_features_options[max_features_idx]

    # Gene 2 -> max_depth: [3, 30] or None (if gene > 0.95, use None)
    if g[2] > 0.95:
        max_depth = None
    else:
        max_depth = int(3 + g[2] * (30 - 3) / 0.95)

    # Gene 3 -> min_samples_split: [2, 20], integer
    min_samples_split = int(2 + g[3] * (20 - 2))

    # Gene 4 -> min_samples_leaf: [1, 10], integer
    min_samples_leaf = int(1 + g[4] * (10 - 1))

    return {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }
```

### Fitness function

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def evaluate_individual(individual, X_train, y_train):
    """Fitness = mean 5-fold CV F1-weighted score."""
    params = decode_individual(individual)
    try:
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_weighted")
        return (scores.mean(),)  # DEAP expects a tuple
    except Exception as e:
        print(f"  [WARN] Individual failed: {e}")
        return (0.0,)  # Penalize invalid configs
```

### DEAP setup — full implementation pattern

```python
import random
from deap import base, creator, tools
import numpy as np

RANDOM_SEED = 42
POP_SIZE = 50
N_GENERATIONS = 30
CX_PROB = 0.7       # Crossover probability
MUT_PROB = 0.2      # Mutation probability
TOURNAMENT_SIZE = 3
N_ELITES = 2        # Number of top individuals preserved each generation

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 1. Define fitness and individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)  # Each gene is a float in [0,1]
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 2. Register operators
toolbox.register("evaluate", evaluate_individual, X_train=X_train, y_train=y_train)
toolbox.register("mate", tools.cxBlend, alpha=0.5)          # Blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

# 3. Clamp genes to [0,1] after crossover/mutation
def clamp(individual):
    for i in range(len(individual)):
        individual[i] = max(0.0, min(1.0, individual[i]))
    return individual

# 4. Evolution loop with logging and population snapshots
def run_evolution():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(N_ELITES)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "min", "max"]

    snapshot_gens = [0, 10, 20, 30]
    snapshots = {}

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    print(logbook.stream)

    if 0 in snapshot_gens:
        snapshots[0] = [decode_individual(ind) for ind in pop]

    for gen in range(1, N_GENERATIONS + 1):
        # Select next generation
        offspring = toolbox.select(pop, len(pop) - N_ELITES)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                clamp(child1)
                clamp(child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                clamp(mutant)
                del mutant.fitness.values

        # Evaluate individuals with invalidated fitness
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalids))
        for ind, fit in zip(invalids, fitnesses):
            ind.fitness.values = fit

        # Elitism: add hall of fame back
        elites = list(map(toolbox.clone, hof.items))
        pop[:] = elites + offspring
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalids), **record)
        print(logbook.stream)

        if gen in snapshot_gens:
            snapshots[gen] = [decode_individual(ind) for ind in pop]

    return pop, logbook, hof, snapshots
```

### Logging and output

After `run_evolution()` completes:

1. Extract best individual from `hof[0]`, decode to params dict.
2. Run 5-fold CV one more time with best params to get per-fold scores (needed for Wilcoxon test).
3. Save `results/gp_best_params.json`:
   ```json
   {
     "best_params": {"n_estimators": 340, "max_features": "sqrt", "...": "..."},
     "best_fitness": 0.927,
     "cv_scores": [0.925, 0.929, 0.924, 0.930, 0.927],
     "raw_genes": [0.64, 0.0, 0.85, 0.33, 0.22],
     "time_seconds": 482.3
   }
   ```
4. Save `results/gp_evolution.csv` from the logbook:
   ```csv
   gen,nevals,avg,std,min,max
   0,50,0.8721,0.0234,0.8012,0.9156
   1,38,0.8834,...
   ```
5. Save `results/gp_population_snapshots.json` — the `snapshots` dict keyed by generation number, each value is a list of 50 decoded param dicts.

---

## Step 5: Final Evaluation (`src/evaluate.py`)

### Input
Load all numpy data from `data/`, load results from `results/baseline_results.json` and `results/gp_best_params.json`.

### Operations

**5A. Train final models on full training set, predict on test set:**

For each of the three methods (Default RF, RandomizedSearchCV best, GP best):
```python
clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

Compute using `sklearn.metrics`:
- `accuracy_score(y_test, y_pred)`
- `precision_score(y_test, y_pred, average="weighted")`
- `recall_score(y_test, y_pred, average="weighted")`
- `f1_score(y_test, y_pred, average="weighted")`

**5B. Build comparison table:**

| Method           | Accuracy | Precision | Recall | F1 (weighted) | CV F1 Mean +/- Std | Time (s) |
|------------------|----------|-----------|--------|---------------|---------------------|----------|

Save as `results/comparison_table.csv` with these exact column names.

**5C. Statistical significance test:**

```python
from scipy.stats import wilcoxon

# GP per-fold scores from results/gp_best_params.json["cv_scores"]
# RS per-fold scores from results/baseline_results.json[1]["cv_scores"]
stat, p_value = wilcoxon(gp_cv_scores, rs_cv_scores)

test_result = {
    "test": "Wilcoxon signed-rank",
    "gp_scores": gp_cv_scores,
    "rs_scores": rs_cv_scores,
    "statistic": float(stat),
    "p_value": float(p_value),
    "significant_at_0.05": p_value < 0.05,
}
# Save to results/statistical_test.json
```

**Important:** If the two score arrays are identical (can happen with very similar methods), `wilcoxon` will raise an error. Catch this and report "no difference detected" with p_value=1.0.

**5D. Generate classification report:**
```python
from sklearn.metrics import classification_report
# Use inverse label mapping to get class names
report = classification_report(y_test, y_pred_gp, target_names=class_names)
# Save to results/classification_report.txt
```

**5E. Save feature importances from GP-optimized model:**
```python
importances = clf_gp.feature_importances_
# Save as results/feature_importances.json: {"feature_name": importance_value, ...}
```

**5F. Generate ALL plots** (details below).

---

## Step 6: Plots to Generate

ALL plots go to `results/`. Use these exact settings on every plot:
```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — CRITICAL for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
})
# Save every plot with: plt.savefig("results/FILENAME.png", dpi=300, bbox_inches="tight")
# Always call plt.close() after saving to free memory
```

### Plot 1: `fitness_evolution.png`
- X-axis: Generation (0 to 30)
- Y-axis: F1 Score
- Two lines: "Best Fitness" (`max` column from logbook) and "Average Fitness" (`avg` column)
- Shaded region for avg +/- std using `plt.fill_between()`
- Title: "GP Fitness Evolution Over Generations"
- Legend, grid on

### Plot 2: `comparison_barplot.png`
- Grouped bar chart using `matplotlib` (NOT seaborn — for full control)
- X-axis: Metric names (Accuracy, Precision, Recall, F1)
- 3 bars per metric group: Default RF, RandomizedSearch, GP-Optimized
- Add value labels on top of each bar formatted to 4 decimal places
- Title: "Performance Comparison Across Tuning Methods"
- Use colors: `#2196F3` (default), `#FF9800` (random search), `#4CAF50` (GP)
- Y-axis range: start from a sensible floor (e.g., 0.8) so differences are visible

### Plot 3: `confusion_matrix.png`
- Heatmap using `seaborn.heatmap()`
- Use `sklearn.metrics.confusion_matrix` for GP-optimized model on test set
- Annotate cells with integer counts, format `fmt="d"`
- Use class names (SEKER, BARBUNYA, etc.) as tick labels — load from `results/label_mapping.json`, invert the mapping to get `{0: "BARBUNYA", 1: "BOMBAY", ...}`
- Title: "Confusion Matrix — GP-Optimized Random Forest"
- Colormap: `Blues`
- Set `xticklabels` rotation to 45 degrees

### Plot 4: `hyperparameter_convergence.png`
- 2x3 subplot grid (5 subplots used, 6th is empty or contains a text legend)
- Each subplot: 4 box plots for generations 0, 10, 20, 30
- Data from `results/gp_population_snapshots.json` — extract each hyperparameter across all 50 individuals at each snapshot generation
- Subplot titles: "n_estimators", "max_features (index)", "max_depth", "min_samples_split", "min_samples_leaf"
- For `max_features`: convert string values to numeric index (sqrt=0, log2=1, 0.3=2, etc.) for the box plot
- For `max_depth`: treat None as 31 (one above max range) for visualization
- Suptitle: "Hyperparameter Distribution Across Generations"
- `plt.tight_layout()`

### Plot 5: `feature_importance.png`
- Horizontal bar chart (`plt.barh`)
- Feature names on y-axis, importance on x-axis
- From `results/feature_importances.json`
- Sorted descending (most important at top)
- Title: "Feature Importance — GP-Optimized Random Forest"
- Color: `#4CAF50`

### Plot 6: `time_comparison.png`
- Simple vertical bar chart
- X-axis: Method names (Default RF, RandomizedSearch, GP)
- Y-axis: Wall-clock time in seconds
- Add value labels on bars
- Title: "Computational Cost Comparison"
- Colors: same as Plot 2

---

## Step 7: Draft Paper Sections (`paper_sections/`)

After all results are generated, create markdown files for each empty section. **Base all content on actual numbers from `results/` files.** Load the JSON/CSV files, read the real values, and embed them in the text. Do NOT invent or hardcode numbers.

### `paper_sections/06_experiments.md`
Write the Experiments / Implementation section covering these sub-sections:

**6.1 Dataset:** Read `data/dataset_source.txt` to determine if real or synthetic data was used. If real: "The Dry Bean dataset from the UCI Machine Learning Repository (Koklu & Ozkan, 2020) was used. It contains 13,611 samples with 16 numeric shape descriptors for seven types of dry beans." Include exact class counts from `results/class_distribution.csv`. If synthetic: note it clearly as a limitation.

**6.2 Model:** "scikit-learn's RandomForestClassifier was used as the base classifier. Five hyperparameters were targeted:" then list the five with their search ranges in a table.

**6.3 Baseline:** "Two baselines: (1) Default RF with scikit-learn defaults (n_estimators=100, max_features='sqrt', max_depth=None, min_samples_split=2, min_samples_leaf=1). (2) RandomizedSearchCV with n_iter=50, 5-fold stratified CV, scoring='f1_weighted'."

**6.4 Optimization Method:** "DEAP library implementing a genetic algorithm: population=50, generations=30, blend crossover (alpha=0.5, p=0.7), Gaussian mutation (sigma=0.1, p=0.2), tournament selection (k=3), elitism (top 2). Each individual: 5 genes in [0,1] decoded to hyperparameter values. Fitness: 5-fold CV weighted F1."

**6.5 Comparison:** "All methods used identical data splits (80/20, stratified, seed=42), CV folds, scoring metric, and random seed."

### `paper_sections/07_results.md`
- Present the comparison table with actual numbers from `results/comparison_table.csv`.
- Describe the fitness evolution curve — which generation convergence occurred (look for where max fitness plateaus).
- Describe confusion matrix patterns — which classes had highest/lowest recall.
- Report Wilcoxon test: statistic value, p-value, whether significant.
- Reference figures: "As shown in Figure 1 (fitness_evolution.png)...", etc.

### `paper_sections/08_discussion.md`
- Did GP outperform RandomSearch? Calculate the absolute and percentage difference in F1.
- Was it statistically significant? Interpret the p-value.
- What hyperparameters did GP converge on? Read from `results/gp_best_params.json` and discuss whether values make domain sense (e.g., more trees = better but diminishing returns, moderate depth avoids overfitting).
- Computational cost: compare times. Is the GP overhead justified by the performance gain?
- Limitation: this is technically a GA, not tree-based GP — the paper title uses "Genetic Programming" broadly following the evolutionary computation umbrella. Reference Koza [5] for the distinction.

### `paper_sections/09_challenges.md`
- Computational cost: each fitness eval = 5-fold CV of RF = 5 RF fits. Population of 50 x 30 generations = up to 7,500 RF fits (less with elitism caching).
- Mixed-type encoding: max_features has string AND float options, requires index-based mapping.
- Overfitting CV score: GA may overfit the 5-fold CV without transferring to held-out test.
- GA parameter sensitivity: results depend on population size, generation count, mutation rate.
- Reproducibility: `n_jobs=-1` parallelism can cause slight non-determinism across runs.

### `paper_sections/10_future_work.md`
- Multi-objective optimization (accuracy vs training time) using NSGA-II in DEAP.
- True tree-based GP for automated feature construction combined with hyperparameter tuning.
- Extend to gradient boosting (XGBoost, LightGBM), SVM, and neural networks.
- Compare against Bayesian optimization (Optuna, Hyperopt, SMAC).
- Test on larger/more diverse datasets from different domains (medical, NLP, image).
- Adaptive evolutionary parameters (self-adapting mutation rates).
- Ensemble of top-N GA individuals instead of single best.

---

## Step 8: Pipeline Runner (`run_all.py`)

Single script that runs everything in order:

```python
#!/usr/bin/env python3
"""Run the complete RF-GP optimization pipeline."""
import subprocess, sys, time

steps = [
    ("Download Dataset",     [sys.executable, "src/download_data.py"]),
    ("Preprocess",           [sys.executable, "src/preprocess.py"]),
    ("Run Baselines",        [sys.executable, "src/baseline.py"]),
    ("Run GP Optimization",  [sys.executable, "src/gp_optimize.py"]),
    ("Evaluate & Plot",      [sys.executable, "src/evaluate.py"]),
]

if __name__ == "__main__":
    total_start = time.time()
    for name, cmd in steps:
        print(f"\n{'='*60}")
        print(f"  STEP: {name}")
        print(f"{'='*60}\n")
        result = subprocess.run(cmd, check=True)
    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE -- Total time: {total:.1f}s")
    print(f"  Results in: results/")
    print(f"  Paper drafts in: paper_sections/")
    print(f"{'='*60}")
```

---

## Shared Constants (`src/utils.py`)

```python
import os, json
import numpy as np

RANDOM_SEED = 42
DATA_DIR = "data"
RESULTS_DIR = "results"
PAPER_DIR = "paper_sections"
TARGET_COLUMN = "Class"
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING_METRIC = "f1_weighted"

# Ensure directories exist
for d in [DATA_DIR, RESULTS_DIR, PAPER_DIR]:
    os.makedirs(d, exist_ok=True)

class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)
    print(f"Saved: {path}")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_data():
    """Load preprocessed train/test splits and metadata."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    feature_names = load_json(os.path.join(DATA_DIR, "feature_names.json"))
    label_mapping = load_json(os.path.join(RESULTS_DIR, "label_mapping.json"))
    return X_train, X_test, y_train, y_test, feature_names, label_mapping
```

---

## Hyperparameter Search Space Reference

This table is the single source of truth. Use it in ALL scripts (baseline, GP, evaluate):

| Hyperparameter     | sklearn param        | Range                                        | Type      | Notes                              |
|--------------------|----------------------|----------------------------------------------|-----------|------------------------------------|
| Number of trees    | `n_estimators`       | [50, 500] step 10                            | int       | More trees = less variance         |
| Features per split | `max_features`       | ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9]        | mixed     | Controls tree diversity            |
| Max tree depth     | `max_depth`          | [3, 30] or None                              | int/None  | None = unlimited depth             |
| Min samples split  | `min_samples_split`  | [2, 20]                                      | int       | Higher = more regularization       |
| Min samples leaf   | `min_samples_leaf`   | [1, 10]                                      | int       | Higher = smoother decision boundary|

---

## Execution Checklist

After the pipeline finishes, verify ALL these files exist. If any is missing, there is a bug:

```
data/dry_bean.csv
data/dataset_source.txt
data/README.md
data/X_train.npy
data/X_test.npy
data/y_train.npy
data/y_test.npy
data/feature_names.json
results/label_mapping.json
results/class_distribution.csv
results/baseline_results.json
results/random_search_cv_results.csv
results/gp_best_params.json
results/gp_evolution.csv
results/gp_population_snapshots.json
results/comparison_table.csv
results/statistical_test.json
results/classification_report.txt
results/feature_importances.json
results/fitness_evolution.png
results/comparison_barplot.png
results/confusion_matrix.png
results/hyperparameter_convergence.png
results/feature_importance.png
results/time_comparison.png
paper_sections/06_experiments.md
paper_sections/07_results.md
paper_sections/08_discussion.md
paper_sections/09_challenges.md
paper_sections/10_future_work.md
```

---

## Code Quality Rules

- Type hints on all function signatures.
- Google-style docstrings on all public functions.
- `if __name__ == "__main__":` guard in every script.
- No hardcoded paths — use constants from `utils.py`.
- Every script must be runnable independently (loads its own inputs from disk).
- Print clear `=== Section Name ===` headers during execution.
- Handle exceptions gracefully with useful error messages.
- Use `matplotlib.use("Agg")` at the top of any file that generates plots (headless compatibility).
- Always `plt.close()` after `plt.savefig()`.
- If `n_jobs=-1` causes issues, catch the error and retry with `n_jobs=1`.
