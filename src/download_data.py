"""Download the Dry Bean dataset from UCI, with a synthetic fallback.

Run directly:
    python src/download_data.py [--refresh-data]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import pandas as pd
import requests

# Allow `python src/download_data.py` from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR  # noqa: E402


UCI_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
CSV_PATH = os.path.join(DATA_DIR, "dry_bean.csv")
SOURCE_FLAG_PATH = os.path.join(DATA_DIR, "dataset_source.txt")
README_PATH = os.path.join(DATA_DIR, "README.md")

README_CONTENT = """# Dataset: Dry Bean Dataset

- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Citation:** Koklu, M. and Ozkan, I.A., 2020. Multiclass classification of dry beans using computer vision and machine learning techniques. Computers and Electronics in Agriculture, 174, 105507.
- **Samples:** 13,611
- **Features:** 16 (all numeric: geometric shape descriptors)
- **Target:** Class (7 bean types: SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON)
- **Task:** Multi-class classification
- **License:** CC BY 4.0
"""


def write_readme() -> None:
    """Write data/README.md if it doesn't already exist."""
    if not os.path.exists(README_PATH):
        with open(README_PATH, "w") as f:
            f.write(README_CONTENT)
        print(f"Wrote: {README_PATH}")


def write_source_flag(source: str) -> None:
    """Record which path produced the dataset (uci_download or synthetic_fallback)."""
    with open(SOURCE_FLAG_PATH, "w") as f:
        f.write(source + "\n")
    print(f"Wrote: {SOURCE_FLAG_PATH} -> {source}")


def download_dry_bean() -> str:
    """Download the Dry Bean dataset from UCI and save as CSV."""
    print(f"Downloading from {UCI_URL} ...")
    resp = requests.get(UCI_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xlsx_files = [f for f in zf.namelist() if f.endswith(".xlsx")]
        if not xlsx_files:
            raise FileNotFoundError("No .xlsx found in ZIP")
        with zf.open(xlsx_files[0]) as f:
            df = pd.read_excel(f, engine="openpyxl")

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols to {CSV_PATH}")
    write_source_flag("uci_download")
    return CSV_PATH


def generate_fallback_dataset() -> str:
    """Generate a synthetic dataset that mimics Dry Bean's structure."""
    from sklearn.datasets import make_classification

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
    df.to_csv(CSV_PATH, index=False)
    print(f"[FALLBACK] Generated synthetic dataset: {len(df)} rows x {len(df.columns)} cols")
    write_source_flag("synthetic_fallback")
    return CSV_PATH


def acquire_dataset(refresh: bool = False) -> str:
    """Ensure the dataset CSV exists. Try UCI first, fall back to synthetic.

    Args:
        refresh: If True, delete any cached CSV / source flag and re-acquire.
    """
    print("=== Step 1: Dataset Acquisition ===")
    write_readme()

    if refresh:
        for p in (CSV_PATH, SOURCE_FLAG_PATH):
            if os.path.exists(p):
                os.remove(p)
                print(f"Removed cached: {p}")

    if os.path.exists(CSV_PATH):
        source = "unknown"
        if os.path.exists(SOURCE_FLAG_PATH):
            with open(SOURCE_FLAG_PATH) as f:
                source = f.read().strip()
        print(f"Dataset already exists at {CSV_PATH} (source: {source}). Skipping.")
        return CSV_PATH

    try:
        return download_dry_bean()
    except Exception as e:
        print(f"[WARN] UCI download failed: {type(e).__name__}: {e}")
        print("[WARN] Falling back to synthetic dataset.")
        return generate_fallback_dataset()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force re-download/regeneration even if cached CSV exists.",
    )
    args = parser.parse_args()
    refresh = args.refresh_data or os.environ.get("REFRESH_DATA") == "1"
    acquire_dataset(refresh=refresh)


if __name__ == "__main__":
    main()
