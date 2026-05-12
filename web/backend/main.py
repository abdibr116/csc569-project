"""FastAPI backend serving experiment results as JSON endpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path

import markdown
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.metrics import confusion_matrix

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/app/results"))
PAPER_DIR = Path(os.environ.get("PAPER_DIR", "/app/paper_sections"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))

app = FastAPI(title="RF-GP Optimization Results")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


@app.get("/api/comparison")
def get_comparison():
    path = RESULTS_DIR / "comparison_table.csv"
    if not path.exists():
        raise HTTPException(404, "comparison_table.csv not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@app.get("/api/baseline")
def get_baseline():
    path = RESULTS_DIR / "baseline_results.json"
    if not path.exists():
        raise HTTPException(404, "baseline_results.json not found")
    return _load_json(path)


@app.get("/api/gp")
def get_gp():
    path = RESULTS_DIR / "gp_best_params.json"
    if not path.exists():
        raise HTTPException(404, "gp_best_params.json not found")
    return _load_json(path)


@app.get("/api/evolution")
def get_evolution():
    path = RESULTS_DIR / "gp_evolution.csv"
    if not path.exists():
        raise HTTPException(404, "gp_evolution.csv not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@app.get("/api/statistical")
def get_statistical():
    path = RESULTS_DIR / "statistical_test.json"
    if not path.exists():
        raise HTTPException(404, "statistical_test.json not found")
    return _load_json(path)


@app.get("/api/features")
def get_features():
    path = RESULTS_DIR / "feature_importances.json"
    if not path.exists():
        raise HTTPException(404, "feature_importances.json not found")
    return _load_json(path)


@app.get("/api/classification")
def get_classification():
    path = RESULTS_DIR / "classification_report.txt"
    if not path.exists():
        raise HTTPException(404, "classification_report.txt not found")
    return {"report": path.read_text()}


@app.get("/api/confusion")
def get_confusion():
    try:
        y_test = np.load(DATA_DIR / "y_test.npy")
        label_mapping = _load_json(RESULTS_DIR / "label_mapping.json")
        gp_data = _load_json(RESULTS_DIR / "gp_best_params.json")
        inv = {v: k for k, v in label_mapping.items()}
        class_names = [inv[i] for i in sorted(inv.keys())]

        X_train = np.load(DATA_DIR / "X_train.npy")
        y_train = np.load(DATA_DIR / "y_train.npy")
        X_test = np.load(DATA_DIR / "X_test.npy")

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(**gp_data["best_params"], random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred).tolist()
        return {"matrix": cm, "labels": class_names}
    except Exception as e:
        raise HTTPException(500, f"Error computing confusion matrix: {e}")


@app.get("/api/config")
def get_config():
    result = {}
    source_path = DATA_DIR / "dataset_source.txt"
    if source_path.exists():
        result["dataset_source"] = source_path.read_text().strip()

    dist_path = RESULTS_DIR / "class_distribution.csv"
    if dist_path.exists():
        df = pd.read_csv(dist_path)
        result["class_distribution"] = df.to_dict(orient="records")
        result["n_samples"] = int(df["full"].sum())
        result["n_classes"] = len(df)

    gp_path = RESULTS_DIR / "gp_best_params.json"
    if gp_path.exists():
        gp = _load_json(gp_path)
        result["ga_config"] = gp.get("config", {})
        result["n_seeds"] = gp.get("n_seeds", 1)
        result["best_seed"] = gp.get("best_seed")

    feat_path = DATA_DIR / "feature_names.json"
    if feat_path.exists():
        result["n_features"] = len(_load_json(feat_path))

    return result


@app.get("/api/paper/{section}")
def get_paper_section(section: str):
    valid = {
        "06": "06_experiments.md",
        "07": "07_results.md",
        "08": "08_discussion.md",
        "09": "09_challenges.md",
        "10": "10_future_work.md",
    }
    if section not in valid:
        raise HTTPException(404, f"Section {section} not found. Valid: {list(valid.keys())}")
    path = PAPER_DIR / valid[section]
    if not path.exists():
        raise HTTPException(404, f"{valid[section]} not found")
    md_text = path.read_text()
    html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    return {"markdown": md_text, "html": html}


@app.get("/api/paper")
def get_paper_sections():
    sections = []
    for num, fname in [("06", "06_experiments.md"), ("07", "07_results.md"),
                       ("08", "08_discussion.md"), ("09", "09_challenges.md"),
                       ("10", "10_future_work.md")]:
        path = PAPER_DIR / fname
        if path.exists():
            md_text = path.read_text()
            html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
            sections.append({"section": num, "filename": fname, "html": html})
    return sections


@app.get("/api/plots/{filename}")
def get_plot(filename: str):
    if not filename.endswith(".png"):
        raise HTTPException(400, "Only PNG files supported")
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/seeds")
def get_seeds():
    gp_path = RESULTS_DIR / "gp_best_params.json"
    if not gp_path.exists():
        return {"seeds": []}
    gp = _load_json(gp_path)
    return {
        "best_seed": gp.get("best_seed"),
        "n_seeds": gp.get("n_seeds", 1),
        "summary": gp.get("all_seeds_summary", []),
        "f1_mean_across_seeds": gp.get("f1_mean_across_seeds"),
        "f1_std_across_seeds": gp.get("f1_std_across_seeds"),
    }
