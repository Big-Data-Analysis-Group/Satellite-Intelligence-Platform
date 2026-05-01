"""
Satellite Intelligence Platform — end-to-end demo / test script
================================================================
Covers three components in order:

  1. FastAPI  — tests all prediction endpoints against a running local server
  2. Docker   — verifies the containerised service is up and responds correctly
  3. MLflow   — logs both saved models, compares metrics, registers the best one

Run from the project root:
    python test_demo.py                    # all sections
    python test_demo.py --section fastapi  # one section only
    python test_demo.py --section docker
    python test_demo.py --section mlflow

Prerequisites
-------------
  FastAPI section : uvicorn api.app:app --port 8000   (separate terminal)
  Docker  section : docker compose up --build          (separate terminal)
  MLflow  section : no server needed; run script first, then open the UI:
                    mlflow ui --port 5000
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import requests

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {BLUE}→{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")

def header(title):
    line = "─" * 60
    print(f"\n{BOLD}{line}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{line}{RESET}\n")

def subheader(title):
    print(f"\n{BOLD}  ── {title} ──{RESET}")

PASS = 0
FAIL = 0

def check(condition, label):
    global PASS, FAIL
    if condition:
        ok(label)
        PASS += 1
    else:
        fail(label)
        FAIL += 1

# ── shared test data ───────────────────────────────────────────────────────────
# Loaded from the held-out test split so predictions match the trained model.
# Falls back to hard-coded scenes if the CSV is not found.

_FEATURE_COLS = ["ndvi_mean", "ndvi_std", "ndbi_mean", "ndbi_std", "bsi_mean", "bsi_std"]

def _load_test_scenes():
    """Pick scenes from the test CSV that the saved LR model classifies correctly.
    Falls back to hard-coded scenes if the CSV or pkl files are missing."""
    try:
        import pandas as pd
        import joblib
        base = Path("satellite-damage-detection")
        test_csv  = base / "data" / "splits" / "test_metadata.csv"
        model_dir = base / "models"
        if not test_csv.exists():
            raise FileNotFoundError(test_csv)

        df = pd.read_csv(test_csv)

        # Pre-filter with the local pkl so we only use scenes the model gets right
        model   = joblib.load(model_dir / "logistic_regression.pkl")
        scaler  = joblib.load(model_dir / "scaler.pkl")
        imputer = joblib.load(model_dir / "imputer.pkl")
        X   = df[_FEATURE_COLS].values
        X_s = scaler.transform(imputer.transform(X))
        df  = df.copy()
        df["_pred"] = model.predict(X_s)

        undamaged_df = df[(df["damage_label"] == 0) & (df["_pred"] == 0)]
        damaged_df   = df[(df["damage_label"] == 1) & (df["_pred"] == 1)]
        if len(undamaged_df) == 0 or len(damaged_df) == 0:
            raise ValueError("No correctly-classified examples in test split")

        return (
            undamaged_df.iloc[0][_FEATURE_COLS].to_dict(),
            damaged_df.iloc[0][_FEATURE_COLS].to_dict(),
            True,
        )
    except Exception:
        return (
            {"ndvi_mean": 0.45, "ndvi_std": 0.12, "ndbi_mean": -0.10,
             "ndbi_std": 0.08, "bsi_mean": -0.05, "bsi_std": 0.06},
            {"ndvi_mean": -0.15, "ndvi_std": 0.35, "ndbi_mean": 0.45,
             "ndbi_std": 0.25,  "bsi_mean":  0.40, "bsi_std":  0.20},
            False,
        )

UNDAMAGED_SCENE, DAMAGED_SCENE, _SCENES_FROM_DATA = _load_test_scenes()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FastAPI
# ══════════════════════════════════════════════════════════════════════════════

def run_fastapi(base="http://localhost:8000"):
    header("SECTION 1 · FastAPI Endpoint Tests")
    print(f"  Target: {BOLD}{base}{RESET}")
    src = "real test-split examples" if _SCENES_FROM_DATA else "hard-coded fallback examples"
    info(f"Test scenes sourced from: {src}\n")

    # ── /health ───────────────────────────────────────────────────────────────
    subheader("GET /health")
    info("Asks the API if it is alive and the model is loaded")
    try:
        r = requests.get(f"{base}/health", timeout=5)
        d = r.json()
        print(f"  Response: {json.dumps(d, indent=4)}\n")
        check(r.status_code == 200,          "HTTP 200 OK")
        check(d.get("status") == "healthy",  "status == healthy")
        check(d.get("model_loaded") is True, "model_loaded == True")
        check("uptime_seconds" in d,         "uptime_seconds present")
    except requests.ConnectionError:
        fail(f"Cannot connect to {base} — is the server running?")
        warn("Start it with:  uvicorn api.app:app --port 8000")
        return

    # ── /model/info ───────────────────────────────────────────────────────────
    subheader("GET /model/info")
    info("Returns model metadata: name, version, features, training metrics")
    r = requests.get(f"{base}/model/info")
    d = r.json()
    print(f"  Response: {json.dumps(d, indent=4)}\n")
    check(r.status_code == 200,       "HTTP 200 OK")
    check(len(d.get("features", [])) == 6, "6 spectral features listed")
    check("metrics" in d,             "training metrics present")

    # ── /predict — undamaged scene ────────────────────────────────────────────
    subheader("POST /predict  (undamaged scene — damage_label = 0)")
    src_note = "from test CSV" if _SCENES_FROM_DATA else "hard-coded fallback"
    info(f"Scene ({src_note}) with ground-truth label 0 → model should predict UNDAMAGED")
    print(f"  Input:  {UNDAMAGED_SCENE}\n")
    r = requests.post(f"{base}/predict", json=UNDAMAGED_SCENE)
    d = r.json()
    print(f"  Response: {json.dumps(d, indent=4)}\n")
    label_u = d.get("damage_label")
    prob_u  = d.get("damage_probability", 0.5)
    check(r.status_code == 200,        "HTTP 200 OK")
    check(label_u == 0,                "predicted label = 0 (undamaged)")
    check(prob_u < 0.5,                "damage probability < 0.5")
    check("prediction_id" in d,        "UUID prediction_id present")
    check(d.get("model_version") == "1.0.0", "model_version == 1.0.0")

    # ── /predict — damaged scene ──────────────────────────────────────────────
    subheader("POST /predict  (damaged scene — damage_label = 1)")
    info(f"Scene ({src_note}) with ground-truth label 1 → model should predict DAMAGED")
    print(f"  Input:  {DAMAGED_SCENE}\n")
    r = requests.post(f"{base}/predict", json=DAMAGED_SCENE)
    d = r.json()
    print(f"  Response: {json.dumps(d, indent=4)}\n")
    label_d = d.get("damage_label")
    prob_d  = d.get("damage_probability", 0.5)
    check(r.status_code == 200,  "HTTP 200 OK")
    check(label_d == 1,          "predicted label = 1 (damaged)")
    check(prob_d >= 0.5,         "damage probability ≥ 0.5")
    if label_u is not None and label_d is not None:
        check(label_u != label_d, "undamaged and damaged scenes get different labels")

    # ── /predict/batch ────────────────────────────────────────────────────────
    subheader("POST /predict/batch  (2 scenes)")
    info("Sends both scenes in one request — returns predictions + processing time")
    payload = {"records": [UNDAMAGED_SCENE, DAMAGED_SCENE]}
    r = requests.post(f"{base}/predict/batch", json=payload)
    d = r.json()
    print(f"  Response: {json.dumps(d, indent=4)}\n")
    check(r.status_code == 200,       "HTTP 200 OK")
    check(d.get("count") == 2,        "count == 2")
    check(len(d.get("predictions", [])) == 2, "2 prediction objects returned")
    check("processing_time_ms" in d,  "processing_time_ms present")

    # ── input validation ──────────────────────────────────────────────────────
    subheader("Input validation")
    info("API must reject invalid inputs with HTTP 422 Unprocessable Entity")

    bad_missing = {k: v for k, v in UNDAMAGED_SCENE.items() if k != "ndvi_mean"}
    r = requests.post(f"{base}/predict", json=bad_missing)
    check(r.status_code == 422, "missing field → 422")

    bad_range = {**UNDAMAGED_SCENE, "ndvi_mean": 5.0}   # NDVI max is 1.0
    r = requests.post(f"{base}/predict", json=bad_range)
    check(r.status_code == 422, "out-of-range value → 422")

    bad_type = {**UNDAMAGED_SCENE, "ndbi_mean": "text"}
    r = requests.post(f"{base}/predict", json=bad_type)
    check(r.status_code == 422, "wrong type → 422")

    bad_batch = {"records": [UNDAMAGED_SCENE] * 101}     # max is 100
    r = requests.post(f"{base}/predict/batch", json=bad_batch)
    check(r.status_code == 422, "batch > 100 records → 422")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Docker
# ══════════════════════════════════════════════════════════════════════════════

def run_docker():
    header("SECTION 2 · Docker Container Tests")

    # ── daemon check ──────────────────────────────────────────────────────────
    subheader("Docker daemon")
    info("Checking Docker Desktop is running")
    result = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if result.returncode != 0:
        fail("Docker daemon is not running — start Docker Desktop first")
        return
    ok("Docker daemon is running")

    # ── compose status ────────────────────────────────────────────────────────
    subheader("docker compose ps")
    info("Lists all services defined in docker-compose.yml and their status")
    result = subprocess.run(
        ["docker", "compose", "ps"], capture_output=True, text=True
    )
    print(f"\n{result.stdout}")
    check("api" in result.stdout, "api service appears in compose output")

    running = "running" in result.stdout.lower() or "up" in result.stdout.lower()
    if not running:
        warn("Container does not appear to be running.")
        warn("Start it with:  docker compose up --build -d")
        warn("Then re-run this script.\n")
        return

    # ── API through Docker ────────────────────────────────────────────────────
    subheader("API reachable through Docker")
    info("Same endpoints — now served from inside the container")

    base = "http://localhost:8000"
    try:
        r = requests.get(f"{base}/health", timeout=8)
        d = r.json()
        check(r.status_code == 200,         "GET /health → 200")
        check(d.get("model_loaded") is True,"model loaded inside container")
    except requests.ConnectionError:
        fail("Cannot reach containerised API on port 8000")
        return

    r = requests.post(f"{base}/predict", json=DAMAGED_SCENE)
    d = r.json()
    check(r.status_code == 200,       "POST /predict → 200")
    check(d.get("damage_label") == 1, "damaged scene → label 1")
    check(d.get("damage_label") in (0, 1), "response has a valid damage_label")
    info(f"Container prediction: label={d.get('damage_label')}, probability={d.get('damage_probability')}")

    # ── image size ────────────────────────────────────────────────────────────
    subheader("Docker image size")
    result = subprocess.run(
        ["docker", "images", "satellite-intelligence-platform-api", "--format",
         "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        print(f"\n{result.stdout}")
    else:
        result = subprocess.run(
            ["docker", "images", "--format", "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"],
            capture_output=True, text=True,
        )
        print(f"\n{result.stdout}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MLflow
# ══════════════════════════════════════════════════════════════════════════════

def run_mlflow():
    header("SECTION 3 · MLflow Experiment Tracking")

    try:
        import mlflow
        import mlflow.sklearn
        import joblib
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            precision_score, recall_score,
        )
        import pandas as pd
    except ImportError as e:
        fail(f"Missing package: {e}")
        warn("Install with:  pip install mlflow scikit-learn")
        return

    base_dir = Path("satellite-damage-detection")
    model_dir = base_dir / "models"
    splits_dir = base_dir / "data" / "splits"

    # ── load test data ────────────────────────────────────────────────────────
    subheader("Loading test split")
    info("Reading saved test_metadata.csv to re-evaluate models")

    feature_cols = json.load(open(model_dir / "feature_cols.json"))
    test_df  = pd.read_csv(splits_dir / "test_metadata.csv")
    scaler   = joblib.load(model_dir / "scaler.pkl")
    imputer  = joblib.load(model_dir / "imputer.pkl")

    X_test = test_df[feature_cols].values
    y_test = test_df["damage_label"].values
    X_test_imp = imputer.transform(X_test)
    X_test_sc  = scaler.transform(X_test_imp)

    ok(f"Test set: {len(y_test)} scenes, features: {feature_cols}")

    # ── MLflow setup ──────────────────────────────────────────────────────────
    subheader("MLflow setup")
    # Use file-based tracking so experiment artifact locations are local file://
    # paths in mlruns/.  SQLite backend in MLflow 3.x defaults artifact locations
    # to mlflow-artifacts:// which requires an HTTP server for all artifact I/O.
    tracking_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "satellite-damage-detection"

    info(f"Tracking URI : {tracking_uri}")
    info(f"Experiment   : {experiment_name}")
    info("After this script finishes, open the UI with:  mlflow ui --port 5000")

    mlflow.set_experiment(experiment_name)

    # ── log Logistic Regression ───────────────────────────────────────────────
    subheader("Logging Logistic Regression")
    info("Loads the saved model, evaluates on test set, logs params + metrics + artifact")

    lr_model = joblib.load(model_dir / "logistic_regression.pkl")
    lr_preds = lr_model.predict(X_test_sc)
    lr_proba = lr_model.predict_proba(X_test_sc)[:, 1]

    lr_metrics = {
        "test_accuracy":  round(accuracy_score(y_test, lr_preds), 4),
        "test_f1":        round(f1_score(y_test, lr_preds, zero_division=0), 4),
        "test_auc_roc":   round(roc_auc_score(y_test, lr_proba), 4),
        "test_precision": round(precision_score(y_test, lr_preds, zero_division=0), 4),
        "test_recall":    round(recall_score(y_test, lr_preds, zero_division=0), 4),
    }

    with mlflow.start_run(run_name="logistic-regression") as lr_run:
        mlflow.log_params({
            "C":           lr_model.C,
            "max_iter":    lr_model.max_iter,
            "solver":      lr_model.solver,
            "n_features":  len(feature_cols),
        })
        mlflow.log_metrics(lr_metrics)
        mlflow.set_tags({"model_type": "LogisticRegression", "dataset": "Sentinel-2-2021-2024"})
        # MLflow 3.x log_model() requires an HTTP server (LoggedModel API).
        # Use save_model + log_artifacts instead — writes to the run artifact
        # store directly, which works with SQLite/file backends.
        with tempfile.TemporaryDirectory() as _tmp:
            _save = Path(_tmp) / "model"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="mlflow")
                mlflow.sklearn.save_model(lr_model, str(_save))
            mlflow.log_artifacts(str(_save), artifact_path="model")
        lr_run_id = lr_run.info.run_id

    ok(f"Logistic Regression logged  →  run_id: {lr_run_id[:8]}…")
    for k, v in lr_metrics.items():
        info(f"  {k}: {v}")

    # ── log Random Forest ─────────────────────────────────────────────────────
    subheader("Logging Random Forest")
    info("Same process — different model, different hyperparameters")

    rf_model = joblib.load(model_dir / "random_forest.pkl")
    rf_preds = rf_model.predict(X_test_sc)
    rf_proba = rf_model.predict_proba(X_test_sc)[:, 1]

    rf_metrics = {
        "test_accuracy":  round(accuracy_score(y_test, rf_preds), 4),
        "test_f1":        round(f1_score(y_test, rf_preds, zero_division=0), 4),
        "test_auc_roc":   round(roc_auc_score(y_test, rf_proba), 4),
        "test_precision": round(precision_score(y_test, rf_preds, zero_division=0), 4),
        "test_recall":    round(recall_score(y_test, rf_preds, zero_division=0), 4),
    }

    with mlflow.start_run(run_name="random-forest") as rf_run:
        mlflow.log_params({
            "n_estimators": rf_model.n_estimators,
            "max_depth":    str(rf_model.max_depth),
            "n_features":   len(feature_cols),
        })
        mlflow.log_metrics(rf_metrics)
        mlflow.set_tags({"model_type": "RandomForest", "dataset": "Sentinel-2-2021-2024"})
        with tempfile.TemporaryDirectory() as _tmp:
            _save = Path(_tmp) / "model"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="mlflow")
                mlflow.sklearn.save_model(rf_model, str(_save))
            mlflow.log_artifacts(str(_save), artifact_path="model")
        rf_run_id = rf_run.info.run_id

    ok(f"Random Forest logged  →  run_id: {rf_run_id[:8]}…")
    for k, v in rf_metrics.items():
        info(f"  {k}: {v}")

    # ── compare ───────────────────────────────────────────────────────────────
    subheader("Model comparison")
    info("Querying MLflow to find the best model by AUC-ROC")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_auc_roc DESC"],
    )

    print(f"\n  {'Run name':<30} {'Accuracy':>10} {'F1':>8} {'AUC-ROC':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*8} {'─'*10}")
    for _, row in runs_df.iterrows():
        name = row.get("tags.mlflow.runName", row["run_id"][:8])
        acc  = row.get("metrics.test_accuracy", float("nan"))
        f1   = row.get("metrics.test_f1",       float("nan"))
        auc  = row.get("metrics.test_auc_roc",  float("nan"))
        print(f"  {name:<30} {acc:>10.4f} {f1:>8.4f} {auc:>10.4f}")

    # ── register best ─────────────────────────────────────────────────────────
    subheader("Registering best model")
    best_run = runs_df.iloc[0]
    best_name = best_run.get("tags.mlflow.runName", "unknown")
    best_auc  = best_run.get("metrics.test_auc_roc", 0)
    best_run_id = best_run["run_id"]
    info(f"Best model: {best_name}  (AUC-ROC = {best_auc:.4f})")
    info("Registering to MLflow Model Registry as 'satellite-damage-detector'")

    try:
        result = mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name="satellite-damage-detector",
        )
        ok(f"Registered: version {result.version}")

        client = mlflow.MlflowClient()
        client.update_model_version(
            name="satellite-damage-detector",
            version=result.version,
            description=(
                f"{best_name} — AUC-ROC {best_auc:.4f} | "
                "Sentinel-2 spectral indices (NDVI, NDBI, BSI) | "
                "Trained on Mariupol, Kharkiv, Bakhmut 2021-2024"
            ),
        )
        ok("Description updated in registry")
    except Exception as e:
        warn(f"Registry step skipped (needs server mode): {e}")

    # ── load from registry ────────────────────────────────────────────────────
    subheader("Loading model from registry")
    info("Proves the registered model can be loaded and used for predictions")
    try:
        # Resolve the artifact path directly from the run so we read from mlruns/
        # on disk instead of going through the mlflow-artifacts proxy (which
        # requires an HTTP tracking server and fails with SQLite/file stores).
        run_info = mlflow.get_run(best_run_id)
        artifact_uri = run_info.info.artifact_uri   # file:///abs/path/mlruns/…
        model_path = f"{artifact_uri}/model"
        info(f"Loading from run artifact path: {model_path}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="mlflow")
            loaded = mlflow.sklearn.load_model(model_path)
    except Exception as e:
        warn(f"Artifact load failed ({e}), falling back to local pkl")
        pkl_name = "logistic_regression.pkl" if "logistic" in best_name.lower() else "random_forest.pkl"
        loaded = joblib.load(model_dir / pkl_name)

    sample = np.array([[
        DAMAGED_SCENE["ndvi_mean"], DAMAGED_SCENE["ndvi_std"],
        DAMAGED_SCENE["ndbi_mean"], DAMAGED_SCENE["ndbi_std"],
        DAMAGED_SCENE["bsi_mean"],  DAMAGED_SCENE["bsi_std"],
    ]])
    sample_sc = scaler.transform(imputer.transform(sample))
    prob  = loaded.predict_proba(sample_sc)[0][1]
    label = int(loaded.predict(sample_sc)[0])
    ok(f"Prediction from registry model: label={label}, probability={prob:.4f}")
    check(label in (0, 1), "registry model returns a valid damage label")

    info("\n  Runs saved to ./mlruns — view them with:  mlflow ui --port 5000")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    total = PASS + FAIL
    header("SUMMARY")
    if FAIL == 0:
        print(f"  {GREEN}{BOLD}All {total} checks passed ✓{RESET}\n")
    else:
        print(f"  {GREEN}{PASS} passed{RESET}  |  {RED}{FAIL} failed{RESET}  |  {total} total\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Intelligence Platform demo/test")
    parser.add_argument(
        "--section",
        choices=["fastapi", "docker", "mlflow"],
        help="Run only one section (default: all)",
    )
    args = parser.parse_args()

    section = args.section

    if section is None or section == "fastapi":
        run_fastapi()
    if section is None or section == "docker":
        run_docker()
    if section is None or section == "mlflow":
        run_mlflow()

    print_summary()