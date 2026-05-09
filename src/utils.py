"""Shared constants, JSON helpers, and data loaders for the RF-GP pipeline."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


RANDOM_SEED: int = 42
DATA_DIR: str = "data"
RESULTS_DIR: str = "results"
PAPER_DIR: str = "paper_sections"
TARGET_COLUMN: str = "Class"
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
SCORING_METRIC: str = "f1_weighted"

for _d in (DATA_DIR, RESULTS_DIR, PAPER_DIR):
    os.makedirs(_d, exist_ok=True)


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj: Any, path: str) -> None:
    """Write `obj` to `path` as pretty-printed JSON, handling numpy types."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)
    print(f"Saved: {path}")


def load_json(path: str) -> Any:
    """Load JSON from `path`."""
    with open(path) as f:
        return json.load(f)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Load preprocessed train/test splits, feature names, and label mapping.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, label_mapping).
    """
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    feature_names = load_json(os.path.join(DATA_DIR, "feature_names.json"))
    label_mapping = load_json(os.path.join(RESULTS_DIR, "label_mapping.json"))
    return X_train, X_test, y_train, y_test, feature_names, label_mapping
