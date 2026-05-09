# RF Hyperparameter Optimization via Genetic Programming

**CSC 569 — Selected Topics in AI, King Saud University, Winter 2026**

This project compares three approaches to Random Forest hyperparameter tuning on the [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (13,611 samples, 16 features, 7 classes):

1. **Default RF** — scikit-learn defaults (the "before" baseline)
2. **RandomizedSearchCV** — 50 random hyperparameter samples with 5-fold CV
3. **Genetic Algorithm (DEAP)** — evolutionary optimization over 30 generations (the "after")

The pipeline downloads/preprocesses data, trains all three methods, evaluates on a held-out test set, generates publication-ready plots, runs a Wilcoxon statistical test, and drafts paper sections 6--10 from real numbers.

---

## Project Structure

```
csc569-project/
├── CLAUDE.md                  # Full project specification
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── run_all.py                 # Single entry point for the entire pipeline
├── src/
│   ├── utils.py               # Shared constants, JSON helpers, data loaders
│   ├── download_data.py       # UCI download with synthetic fallback
│   ├── preprocess.py          # Clean, encode, scale, split
│   ├── baseline.py            # Default RF + RandomizedSearchCV
│   ├── gp_optimize.py         # DEAP genetic algorithm optimization
│   ├── evaluate.py            # Test-set evaluation, comparison table, 6 plots
│   └── draft_paper.py         # Auto-generate paper sections 06--10
├── data/
│   ├── README.md              # Dataset citation and description
│   ├── dry_bean.csv           # Raw dataset (downloaded or generated)
│   ├── dataset_source.txt     # "uci_download" or "synthetic_fallback"
│   ├── X_train.npy            # Preprocessed training features
│   ├── X_test.npy             # Preprocessed test features
│   ├── y_train.npy            # Training labels (encoded)
│   ├── y_test.npy             # Test labels (encoded)
│   └── feature_names.json     # List of 16 feature column names
├── results/                   # All outputs (CSV, JSON, PNG)
│   ├── comparison_table.csv   # Main results with delta columns
│   ├── baseline_results.json  # Default RF + RS results
│   ├── gp_best_params.json    # GP best config, CV scores, timing
│   ├── gp_evolution.csv       # Per-generation fitness stats
│   ├── statistical_test.json  # Wilcoxon signed-rank test
│   ├── classification_report.txt
│   ├── feature_importances.json
│   ├── fitness_evolution.png
│   ├── comparison_barplot.png
│   ├── confusion_matrix.png
│   ├── hyperparameter_convergence.png
│   ├── feature_importance.png
│   └── time_comparison.png
└── paper_sections/            # Auto-drafted markdown for sections 6--10
    ├── 06_experiments.md
    ├── 07_results.md
    ├── 08_discussion.md
    ├── 09_challenges.md
    └── 10_future_work.md
```

---

## Setup

### Option A: Docker Compose (recommended)

No Python setup needed — just Docker.

```bash
# Full run (pop=50, gen=30, ~3-4 hours)
docker compose up --build

# Quick smoke run (~5 minutes)
POP_SIZE=8 GENERATIONS=3 docker compose up --build

# Custom configuration
POP_SIZE=20 GENERATIONS=10 SEED=123 docker compose up --build

# Force re-download the dataset
docker compose run pipeline --pop-size 50 --generations 30 --refresh-data
```

Results appear in `data/`, `results/`, and `paper_sections/` on your host (mounted as volumes).

### Option B: Local Python

#### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Full pipeline (recommended)

Runs all 6 steps sequentially: download, preprocess, baselines, GP optimization, evaluation, and paper drafting.

```bash
python run_all.py
```

Default settings: `--pop-size 50 --generations 30 --seed 42`. This takes approximately **3--4 hours** on a modern machine (dominated by the GP step).

### Quick smoke run

Validate the pipeline end-to-end in ~5 minutes with a smaller GP configuration:

```bash
python run_all.py --pop-size 8 --generations 3
```

### All CLI flags for `run_all.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--pop-size N` | 50 | GA population size |
| `--generations G` | 30 | Number of GA generations |
| `--seed S` | 42 | Random seed for reproducibility |
| `--refresh-data` | off | Force re-download/regenerate the dataset |

**Examples:**

```bash
# Medium-sized run (~30 min)
python run_all.py --pop-size 20 --generations 10

# Full run with a different seed
python run_all.py --seed 123

# Force re-download the dataset from UCI
python run_all.py --refresh-data

# Combine flags
python run_all.py --pop-size 30 --generations 15 --refresh-data --seed 7
```

---

## Running Individual Steps

Each script can be run independently (loads its inputs from disk). This is useful for re-running a single step without repeating the entire pipeline.

### Step 1: Download dataset

```bash
python src/download_data.py                # Uses cached data if available
python src/download_data.py --refresh-data # Force re-download
```

- Tries the UCI URL first; falls back to a synthetic dataset on failure
- Writes `data/dry_bean.csv` and `data/dataset_source.txt`

### Step 2: Preprocess

```bash
python src/preprocess.py
```

- Requires: `data/dry_bean.csv`
- Produces: `data/X_train.npy`, `data/X_test.npy`, `data/y_train.npy`, `data/y_test.npy`, `data/feature_names.json`, `results/label_mapping.json`, `results/class_distribution.csv`

### Step 3: Baselines (Default RF + RandomizedSearchCV)

```bash
python src/baseline.py
```

- Requires: preprocessed numpy files in `data/`
- Produces: `results/baseline_results.json`, `results/random_search_cv_results.csv`

### Step 4: GP optimization

```bash
python src/gp_optimize.py                              # Full run (pop=50, gen=30)
python src/gp_optimize.py --pop-size 10 --generations 5 # Quick test
```

All CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--pop-size` | 50 | Population size |
| `--generations` | 30 | Number of generations |
| `--cx-prob` | 0.7 | Crossover probability |
| `--mut-prob` | 0.2 | Mutation probability |
| `--n-elites` | 2 | Elitism count |
| `--cx-alpha` | 0.5 | Blend crossover alpha |
| `--mut-sigma` | 0.1 | Gaussian mutation sigma |
| `--mut-indpb` | 0.2 | Per-gene mutation probability |
| `--tournament-size` | 3 | Tournament selection size |
| `--seed` | 42 | Random seed |

- Requires: preprocessed numpy files in `data/`
- Produces: `results/gp_best_params.json`, `results/gp_evolution.csv`, `results/gp_population_snapshots.json`

### Step 5: Evaluate and plot

```bash
python src/evaluate.py
```

- Requires: `results/baseline_results.json`, `results/gp_best_params.json`, preprocessed data
- Produces: `results/comparison_table.csv` (with before/after delta columns), `results/statistical_test.json`, `results/classification_report.txt`, `results/feature_importances.json`, and all 6 PNG plots

### Step 6: Draft paper sections

```bash
python src/draft_paper.py
```

- Requires: all files in `results/` and `data/dataset_source.txt`
- Produces: `paper_sections/06_experiments.md` through `paper_sections/10_future_work.md`
- All numbers in the drafts are read from the actual result files, not hardcoded

---

## Hyperparameter Search Space

These ranges are shared across RandomizedSearchCV and the GA:

| Hyperparameter | sklearn param | Range | Type |
|----------------|--------------|-------|------|
| Number of trees | `n_estimators` | [50, 500] step 10 | int |
| Features per split | `max_features` | sqrt, log2, 0.3, 0.5, 0.7, 0.9 | mixed |
| Max tree depth | `max_depth` | [3, 30] or None | int/None |
| Min samples to split | `min_samples_split` | [2, 20] | int |
| Min samples per leaf | `min_samples_leaf` | [1, 10] | int |

---

## Outputs

### Comparison table (`results/comparison_table.csv`)

Contains per-method test-set metrics (Accuracy, Precision, Recall, F1), CV statistics, wall-clock time, and explicit delta columns showing each method's improvement over the Default RF baseline.

### Plots (all in `results/`)

| File | Description |
|------|-------------|
| `fitness_evolution.png` | Best and average fitness over GA generations with std shading |
| `comparison_barplot.png` | Grouped bar chart of all metrics across all 3 methods |
| `confusion_matrix.png` | Heatmap for the GP-optimized model on the test set |
| `hyperparameter_convergence.png` | Box plots showing how each hyperparameter's distribution narrows over generations |
| `feature_importance.png` | Horizontal bar chart of GP-optimized model's feature importances |
| `time_comparison.png` | Wall-clock cost comparison across the 3 methods |

### Paper sections (`paper_sections/`)

Auto-generated markdown drafts for sections 6--10 of the research paper, with all numbers populated from actual experimental results.

---

## Dataset

**Primary:** [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) from UCI (Koklu & Ozkan, 2020). 13,611 samples, 16 numeric shape descriptors, 7 bean classes.

**Fallback:** If the UCI download fails (network restrictions, URL changes), a synthetic dataset with matching structure is generated using `sklearn.datasets.make_classification`. The file `data/dataset_source.txt` records which source was used, and the paper drafts adapt their wording accordingly.

---

## Reproducibility

- All random operations use `random_state=42` (or the `--seed` flag value)
- Data split: 80/20 stratified
- CV: 5-fold stratified, scoring on `f1_weighted`
- `n_jobs=-1` enables parallelism but can introduce tiny floating-point non-determinism (< 0.001 F1 variation between runs)

---

## Citation

If using the Dry Bean dataset:

> Koklu, M. and Ozkan, I.A., 2020. Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.
