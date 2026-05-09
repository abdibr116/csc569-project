"""Clean, encode, scale, and split the Dry Bean dataset.

Run directly:
    python src/preprocess.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (  # noqa: E402
    DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    save_json,
)


CSV_PATH = os.path.join(DATA_DIR, "dry_bean.csv")


def main() -> None:
    print("=== Step 2: Preprocessing ===")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Missing {CSV_PATH}. Run src/download_data.py first."
        )

    df = pd.read_csv(CSV_PATH)
    print(f"Shape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Head:\n{df.head()}")

    # Missing values
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    if total_missing == 0:
        print("Missing values: none.")
    else:
        print(f"Missing values:\n{missing[missing > 0]}")
        n_rows = len(df)
        for col in missing[missing > 0].index:
            frac = missing[col] / n_rows
            if frac < 0.05:
                before = len(df)
                df = df.dropna(subset=[col])
                print(f"  Dropped {before - len(df)} rows missing in '{col}' (<5%).")
            else:
                med = df[col].median()
                df[col] = df[col].fillna(med)
                print(f"  Imputed '{col}' with median={med} (>=5%).")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV.")

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    # Encode any non-numeric feature columns (defensive — Dry Bean has none)
    for col in feature_cols:
        if df[col].dtype == object:
            print(f"  [WARN] Non-numeric feature '{col}' — applying LabelEncoder.")
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Encode target
    le = LabelEncoder()
    y_full = le.fit_transform(df[TARGET_COLUMN].astype(str))
    label_mapping = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
    save_json(label_mapping, os.path.join(RESULTS_DIR, "label_mapping.json"))

    X_full = df[feature_cols].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_full,
    )

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Class distributions
    def counts(arr: np.ndarray) -> dict[str, int]:
        unique, c = np.unique(arr, return_counts=True)
        return {le.classes_[int(u)]: int(n) for u, n in zip(unique, c)}

    dist_df = pd.DataFrame({
        "full": pd.Series(counts(y_full)),
        "train": pd.Series(counts(y_train)),
        "test": pd.Series(counts(y_test)),
    }).fillna(0).astype(int)
    dist_df.index.name = "class"
    dist_df.to_csv(os.path.join(RESULTS_DIR, "class_distribution.csv"))
    print(f"Saved: {os.path.join(RESULTS_DIR, 'class_distribution.csv')}")

    # Save numpy splits + feature names
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    save_json(feature_cols, os.path.join(DATA_DIR, "feature_names.json"))

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape:  X={X_test.shape}, y={y_test.shape}")
    print(f"Class distribution:\n{dist_df}")


if __name__ == "__main__":
    main()
