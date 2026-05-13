"""FastAPI backend serving experiment results and experiment control."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import docker
import markdown
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.metrics import confusion_matrix

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/app/results"))
PAPER_DIR = Path(os.environ.get("PAPER_DIR", "/app/paper_sections"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
HOST_PROJECT_DIR = os.environ.get("HOST_PROJECT_DIR", "/data/src/rana/csc569-project")
PIPELINE_IMAGE = os.environ.get("PIPELINE_IMAGE", "csc569-project-pipeline")
EXPERIMENT_CONTAINER = "csc569-experiment"
PROGRESS_FILE = RESULTS_DIR / ".experiment_progress.jsonl"

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


# ── Experiment Control ──────────────────────────────────────────────


_docker_client = None


def _get_docker() -> docker.DockerClient:
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


def _find_experiment_container() -> Any | None:
    try:
        return _get_docker().containers.get(EXPERIMENT_CONTAINER)
    except docker.errors.NotFound:
        return None
    except docker.errors.DockerException:
        return None


def _parse_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {}
    records = []
    for line in PROGRESS_FILE.read_text().strip().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not records:
        return {}

    config: dict = {}
    steps_completed: list[str] = []
    current_step: str | None = None
    current_seed: int | None = None
    experiment_status: str | None = None
    experiment_error: str | None = None
    total_time: float | None = None
    gen_records: list[dict] = []

    for r in records:
        rtype = r.get("type")
        if rtype == "experiment":
            experiment_status = r.get("status")
            if "config" in r:
                config = r["config"]
            if "total_time_s" in r:
                total_time = r["total_time_s"]
            if "error" in r:
                experiment_error = r["error"]
        elif rtype == "step":
            if r["status"] == "running":
                current_step = r["step"]
                current_seed = r.get("seed")
            elif r["status"] == "completed":
                key = r["step"]
                if r.get("seed"):
                    key = f"{r['step']}_seed_{r['seed']}"
                steps_completed.append(key)
                current_step = None
        elif rtype == "generation":
            gen_records.append(r)

    return {
        "config": config,
        "experiment_status": experiment_status,
        "experiment_error": experiment_error,
        "steps_completed": steps_completed,
        "current_step": current_step,
        "current_seed": current_seed,
        "gen_records": gen_records,
        "total_time": total_time,
    }


def _compute_eta(gen_records: list[dict], config: dict) -> dict:
    if not gen_records or not config:
        return {"estimated_remaining_s": None, "estimated_remaining_human": "calculating..."}

    seeds = config.get("seeds", [42])
    total_gen = config.get("generations", 30)
    total_work = len(seeds) * total_gen

    completed_gens = 0
    seeds_done: set[int] = set()
    for r in gen_records:
        if r["gen"] == r["total_gen"]:
            seeds_done.add(r["seed"])
    for seed in seeds:
        seed_gens = [r for r in gen_records if r["seed"] == seed]
        if seed in seeds_done:
            completed_gens += total_gen
        elif seed_gens:
            completed_gens += max(r["gen"] for r in seed_gens)

    gen_times: list[float] = []
    by_seed: dict[int, list[dict]] = {}
    for r in gen_records:
        by_seed.setdefault(r["seed"], []).append(r)
    for seed, gens in by_seed.items():
        gens_sorted = sorted(gens, key=lambda g: g["gen"])
        for i in range(1, len(gens_sorted)):
            dt = gens_sorted[i]["elapsed_s"] - gens_sorted[i - 1]["elapsed_s"]
            if dt > 0:
                gen_times.append(dt)

    if not gen_times:
        return {"estimated_remaining_s": None, "estimated_remaining_human": "calculating...",
                "completed_gens": completed_gens, "total_gens": total_work}

    avg_gen_time = sum(gen_times) / len(gen_times)
    remaining = max(0, (total_work - completed_gens) * avg_gen_time + 60)

    return {
        "estimated_remaining_s": remaining,
        "estimated_remaining_human": _fmt_duration(remaining),
        "avg_gen_time_s": avg_gen_time,
        "completed_gens": completed_gens,
        "total_gens": total_work,
    }


def _fmt_duration(s: float) -> str:
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s // 60)}m {int(s % 60)}s"
    return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"


class ExperimentConfig(BaseModel):
    pop_size: int = Field(default=50, ge=2, le=200)
    generations: int = Field(default=15, ge=1, le=100)
    seeds: list[int] = Field(default=[42, 123, 456], min_length=1, max_length=10)


@app.get("/api/experiment/status")
def experiment_status():
    container = _find_experiment_container()
    progress = _parse_progress()

    if container and container.status == "running":
        gen_records = progress.get("gen_records", [])
        latest_gen = gen_records[-1] if gen_records else None
        eta = _compute_eta(gen_records, progress.get("config", {}))
        return {
            "status": "running",
            "container_id": container.short_id,
            "config": progress.get("config", {}),
            "current_step": progress.get("current_step"),
            "current_seed": progress.get("current_seed"),
            "steps_completed": progress.get("steps_completed", []),
            "gp_progress": {
                "current_gen": latest_gen["gen"] if latest_gen else 0,
                "total_gen": latest_gen["total_gen"] if latest_gen else 0,
                "fitness_max": latest_gen["fitness_max"] if latest_gen else None,
                "fitness_avg": latest_gen["fitness_avg"] if latest_gen else None,
                "current_seed": latest_gen["seed"] if latest_gen else None,
            } if latest_gen else None,
            "eta": eta,
        }

    if progress.get("experiment_status") == "completed":
        return {
            "status": "completed",
            "config": progress.get("config", {}),
            "total_time_s": progress.get("total_time"),
            "total_time_human": _fmt_duration(progress["total_time"]) if progress.get("total_time") else None,
        }

    if progress.get("experiment_status") == "failed":
        return {
            "status": "failed",
            "config": progress.get("config", {}),
            "error": progress.get("experiment_error"),
        }

    if container and container.status == "exited":
        exit_code = container.attrs["State"].get("ExitCode", -1)
        logs = container.logs(tail=20).decode(errors="replace")
        try:
            container.remove()
        except Exception:
            pass
        if exit_code != 0:
            return {"status": "failed", "error": f"Container exited with code {exit_code}", "logs": logs}
        return {"status": "completed", "config": progress.get("config", {})}

    return {"status": "idle"}


@app.post("/api/experiment/start")
def start_experiment(config: ExperimentConfig):
    existing = _find_experiment_container()
    if existing and existing.status == "running":
        raise HTTPException(409, "An experiment is already running")
    if existing:
        try:
            existing.remove(force=True)
        except Exception:
            pass

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    cmd_args = [
        "--pop-size", str(config.pop_size),
        "--generations", str(config.generations),
        "--seeds",
    ] + [str(s) for s in config.seeds] + [
        "--progress-file", "results/.experiment_progress.jsonl",
    ]

    try:
        client = _get_docker()
        container = client.containers.run(
            PIPELINE_IMAGE,
            command=cmd_args,
            name=EXPERIMENT_CONTAINER,
            volumes={
                f"{HOST_PROJECT_DIR}/data": {"bind": "/app/data", "mode": "rw"},
                f"{HOST_PROJECT_DIR}/results": {"bind": "/app/results", "mode": "rw"},
                f"{HOST_PROJECT_DIR}/paper_sections": {"bind": "/app/paper_sections", "mode": "rw"},
            },
            detach=True,
            remove=False,
        )
        return {"status": "started", "container_id": container.short_id}
    except docker.errors.ImageNotFound:
        raise HTTPException(400, f"Pipeline image '{PIPELINE_IMAGE}' not found. Build it first.")
    except docker.errors.APIError as e:
        raise HTTPException(500, f"Docker error: {e}")


@app.post("/api/experiment/stop")
def stop_experiment():
    container = _find_experiment_container()
    if not container:
        raise HTTPException(404, "No experiment container found")
    try:
        container.stop(timeout=10)
        container.remove()
    except Exception:
        pass
    return {"status": "stopped"}


@app.delete("/api/experiment")
def cleanup_experiment():
    container = _find_experiment_container()
    if container:
        try:
            container.remove(force=True)
        except Exception:
            pass
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
    return {"status": "cleaned"}
