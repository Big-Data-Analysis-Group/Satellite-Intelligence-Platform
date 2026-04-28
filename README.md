# Satellite Damage Detection - Ukraine Conflict Analysis

Automated damage assessment using Sentinel-2 satellite imagery to detect and classify infrastructure damage across Ukrainian conflict zones (Mariupol, Kharkiv, Bakhmut).

## Project Structure

```
Satellite-Intelligence-Platform/
├── satellite-damage-detection/
│   ├── notebooks/
│   │   ├── 1-satellite_data_cleaning.ipynb       # Data download, cloud masking, spectral indices
│   │   ├── 2-feature_engineering.ipynb            # Patch/scene dataset preparation and splits
│   │   ├── 2b-rich_features.ipynb                 # Optional: GLCM texture & distribution features
│   │   ├── 3-model_training.ipynb                 # Train LR, RF, MLP, and ResNet-50
│   │   ├── 4-evaluation.ipynb                     # Full evaluation, damage maps, reports
│   │   └── Satellite Intelligence Platform Merged.ipynb  # Combined single-file pipeline
│   ├── src/               # Python source modules
│   ├── data/              # Data directory (raw, processed, patches, splits)
│   ├── models/            # Trained model files and preprocessing artifacts
│   ├── results/           # Outputs: metrics, predictions, plots, damage maps
│   └── scripts/           # Standalone executable scripts
├── api/
│   ├── app.py             # FastAPI prediction service
│   ├── test_app.py        # Pytest unit tests for the API (11 tests)
│   ├── requirements.txt   # API-specific dependencies
│   └── __init__.py
├── dashboard.py           # Streamlit visualisation dashboard (5 tabs)
├── test_demo.py           # End-to-end demo/test script (FastAPI + Docker + MLflow)
├── Dockerfile             # Container image for the API service
├── docker-compose.yml     # Single-service compose configuration
└── .dockerignore          # Excludes raw data and notebooks from the image
```

## Setup

1. **Clone the repository and create a virtual environment:**
   ```bash
   git clone <repo-url>
   cd Satellite-Intelligence-Platform
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install notebook + dashboard dependencies** (from project root):
   ```bash
   pip install jupyter streamlit plotly pandas scikit-learn mlflow requests
   ```

3. **Install API dependencies:**
   ```bash
   pip install -r api/requirements.txt
   ```

4. **Set up data directories** (Windows PowerShell):
   ```powershell
   New-Item -ItemType Directory -Force satellite-damage-detection/data/raw,
     satellite-damage-detection/data/processed,
     satellite-damage-detection/data/patches,
     satellite-damage-detection/data/splits,
     satellite-damage-detection/models,
     satellite-damage-detection/results/metrics,
     satellite-damage-detection/results/predictions,
     satellite-damage-detection/results/plots,
     satellite-damage-detection/results/damage_maps
   ```

## Generated Files (not committed)

The following are produced at runtime and excluded by `.gitignore`:

| Path | What generates it |
|---|---|
| `satellite-damage-detection/models/` | Notebooks 3–4 (trained model pkl files) |
| `satellite-damage-detection/results/` | Notebook 4 (metrics, plots, damage maps) |
| `satellite-damage-detection/data/` | Notebooks 1–2 (raw imagery, processed CSVs) |
| `mlruns/` | `test_demo.py --section mlflow` (MLflow run data) |
| `mlartifacts/` | MLflow artifact store |
| `mlflow.db` | MLflow SQLite backend (if used) |
| `.venv/` | `pip install` |

## Workflow

### Notebook 1 — Data Cleaning (`1-satellite_data_cleaning.ipynb`)

- Connects to the [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com) STAC API
- Downloads Sentinel-2 Level-2A imagery for **Mariupol**, **Kharkiv**, and **Bakhmut** across **14 quarters** (Q1 2021 – Q2 2024)
- Bands downloaded: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR1), B12 (SWIR2), SCL
- Applies cloud masking using the Scene Classification Layer (retains SCL classes 4, 5, and 6 only)
- Computes spectral indices per scene: **NDVI**, **NDBI**, **BSI**
- Saves cleaned `.npy` arrays and a summary CSV (`data/processed/sentinel2_clean_2021_2024.csv`)
- Produces a temporal grid of RGB images and an animated GIF for Mariupol

**Output:** 41 scenes (1 failed due to API size limit), average cloud cover 10.6%, average valid pixels 53.1%

### Notebook 2 — Feature Engineering (`2-feature_engineering.ipynb`)

Configurable pipeline with two key settings:

| Setting | Options | Effect |
|---|---|---|
| `TEMPORAL_RESOLUTION` | `yearly` / `quarterly` | Aggregate scenes per year or keep quarterly granularity |
| `USE_PATCHES` | `True` / `False` | Slice scenes into 128×128 patches (enables ResNet-50) or use scene-level features |

- Assigns damage labels: **0** = 2021 pre-conflict baseline, **1** = 2022–2024 conflict period
- Uses group-based splits (city+year groups kept together) to prevent data leakage
- Default output in yearly/scene-level mode: 12 rows, 27 feature columns
- Saves train/val/test splits to `data/splits/`

### Notebook 2b — Rich Feature Extraction (`2b-rich_features.ipynb`) *(optional)*

Enriches the scene-level splits with **38 additional features** before model training:

| Category | Features | Count |
|---|---|---|
| GLCM texture (NIR band) | Contrast, dissimilarity, homogeneity, energy, correlation, ASM | 6 |
| Quadrant spatial variance | NDVI/NDBI/BSI std per quadrant | 6 |
| Robust distribution stats | IQR, skewness, kurtosis, P10, P90 for each index | 15 |
| Raw band statistics | Mean and std of RED, NIR, SWIR1, SWIR2 | 8 |
| Damage composites | Damage score, NDVI/NDBI ratio, vegetation loss proxy | 3 |

Overwrites the split CSVs in-place so Notebook 3 picks them up automatically. Requires `USE_PATCHES = False` in Notebook 2.

### Notebook 3 — Model Training (`3-model_training.ipynb`)

Trains four classifiers and auto-detects patch vs scene-level mode:

| Model | Mode | Notes |
|---|---|---|
| Logistic Regression | Both | Linear baseline, `max_iter=1000` |
| Random Forest | Both | 100 trees, provides feature importances |
| MLP | Both | 2 hidden layers (64→32), dropout, 50 epochs, Adam |
| ResNet-50 | Patch only | 6-band input layer, cosine annealing LR, early stopping |

ResNet-50 is adapted for Sentinel-2 by replacing the first convolutional layer to accept 6 spectral bands instead of 3 RGB channels.

**Model comparison results (test set):**

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| MLP *(best)* | 0.9337 | 0.9337 | 0.9803 |
| Logistic Regression | 0.9100 | 0.9121 | 0.9742 |
| Random Forest | 0.7488 | 0.7877 | 0.9398 |
| ResNet-50 | 0.4672 | 0.6368 | 0.9026 |

Saves trained model files to `models/` and test predictions to `results/predictions/test_predictions.csv`.

### Notebook 4 — Evaluation (`4-evaluation.ipynb`)

Comprehensive evaluation of all trained models:

- Full classification reports (precision, recall, F1, AUC) for all models
- Confusion matrices
- ROC and Precision-Recall curves
- **Per-city performance** (Bakhmut, Kharkiv, Mariupol)
- **Damage maps** showing predicted damage probability over time per city
- UNOSAT-style ground truth comparison by period
- Spectral index distribution analysis (damaged vs undamaged)
- Threshold sensitivity analysis (optimal threshold: 0.90 for MLP)
- Final summary JSON (`results/metrics/final_summary.json`)

**Per-city results (MLP, full dataset):**

| City | F1 | AUC-ROC |
|---|---|---|
| Kharkiv | 1.000 | 1.000 |
| Bakhmut | 0.981 | 1.000 |
| Mariupol | 0.955 | 0.961 |

## Interactive Demo Components

### Streamlit Dashboard

Visualises spectral trends, model performance, per-city metrics, and damage probability distributions.

```bash
# From the project root
streamlit run dashboard.py
```

Opens automatically at **http://localhost:8501**. Five tabs:

| Tab | What you see |
|---|---|
| Spectral Trends | NDVI / NDBI / BSI over time per city, with a conflict-onset marker |
| Model Comparison | Accuracy, F1, AUC-ROC bar chart across all four models |
| City Performance | Per-city F1 and AUC breakdown for each model |
| Damage Prediction | Histogram of damage probabilities and prediction scatter |
| Summary | Final metrics table from `final_summary.json` |

**Prerequisites:** model artifacts and result files must exist (run notebooks 1–4 first, or use the merged notebook).

---

### FastAPI Prediction Service

A REST API that accepts scene-level spectral features and returns a damage classification.

```bash
# Start the server (from the project root)
uvicorn api.app:app --reload --port 8000
```

API available at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

Key endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | API info and version |
| `GET` | `/health` | Liveness check (200 = ready) |
| `GET` | `/model/info` | Feature list and model metadata |
| `POST` | `/predict` | Single-scene damage prediction |
| `POST` | `/predict/batch` | Up to 100 scenes in one request |

**PowerShell — single prediction:**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"ndvi_mean":0.45,"ndvi_std":0.12,"ndbi_mean":-0.10,"ndbi_std":0.08,"bsi_mean":-0.05,"bsi_std":0.06}'
```

**curl.exe — single prediction:**
```bash
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"ndvi_mean\":0.45,\"ndvi_std\":0.12,\"ndbi_mean\":-0.10,\"ndbi_std\":0.08,\"bsi_mean\":-0.05,\"bsi_std\":0.06}"
```

Response fields: `prediction_id` (UUID), `damage_label` (0/1), `damage_label_text` ("Undamaged"/"Damaged"), `damage_probability`, `model_version`.

**Run the API unit tests:**
```bash
pytest api/test_app.py -v
```

---

### Docker Compose (Containerised API)

Builds and runs the FastAPI service inside a Docker container — no local Python environment needed.

```bash
# Build and start (first run downloads the base image)
docker compose up --build

# Subsequent runs (image already built)
docker compose up

# Stop and remove containers
docker compose down
```

The service is identical to the local API, also on **http://localhost:8000**. The container copies only the model artifacts (`satellite-damage-detection/models/`) — no raw data or notebooks are included.

**Check container status:**
```bash
docker compose ps
```

The healthcheck polls `/health` every 30 s; status shows `healthy` once the model loads (~10 s).

---

### End-to-End Demo Script (`test_demo.py`)

A single script that validates all three components with colour-coded pass/fail output.

```bash
# All three sections in sequence
python test_demo.py

# Individual sections (FastAPI server must be running for fastapi/docker sections)
python test_demo.py --section fastapi   # tests local API on port 8000
python test_demo.py --section docker    # tests Docker container on port 8000
python test_demo.py --section mlflow    # logs models, compares runs, registers best
```

What each section does:

**FastAPI section** — hits every endpoint with a healthy scene and a damaged scene, checks response fields, tests batch prediction, and verifies that invalid inputs are rejected (400 errors).

**Docker section** — same endpoint tests as FastAPI but targets the containerised service. Confirms the container environment is identical to local.

**MLflow section** — loads `logistic_regression.pkl` and `random_forest.pkl` from disk, re-evaluates them on the held-out test split, logs parameters/metrics/artifacts to MLflow, compares run results, registers the winning model to the Model Registry, then loads it back and runs an inference check.

**Prerequisites per section:**

| Section | Requirement |
|---|---|
| fastapi | `uvicorn api.app:app --port 8000` running in a separate terminal |
| docker | `docker compose up --build` running in a separate terminal |
| mlflow | Model `.pkl` files in `satellite-damage-detection/models/` (no server needed during the run) |

After `--section mlflow` completes, open the tracking UI:
```bash
mlflow ui --port 5000
```
Browse to `http://localhost:5000` to compare runs and inspect the registered model.

---

## Key Files

**Notebooks:**
- `notebooks/1-satellite_data_cleaning.ipynb` — Data acquisition and preprocessing
- `notebooks/2-feature_engineering.ipynb` — Dataset preparation and splitting
- `notebooks/2b-rich_features.ipynb` — Optional texture and distribution feature enrichment
- `notebooks/3-model_training.ipynb` — Model training and comparison
- `notebooks/4-evaluation.ipynb` — Evaluation, damage maps, and reporting
- `notebooks/Satellite Intelligence Platform Merged.ipynb` — Full pipeline in one file

**Source modules:**
- `src/data.py` — Load and preprocess Sentinel-2 imagery
- `src/features.py` — Spectral index calculations
- `src/model.py` — Model architectures
- `src/train.py` — Training utilities
- `src/utils.py` — Helper functions

**Model artifacts (generated after training):**
- `models/logistic_regression.pkl` — Trained Logistic Regression
- `models/random_forest.pkl` — Trained Random Forest
- `models/neural_network.pth` — Trained MLP weights
- `models/resnet50_best.pth` — Trained ResNet-50 weights (patch mode only)
- `models/feature_cols.json`, `models/imputer.pkl`, `models/scaler.pkl` — Preprocessing artifacts

## Data Sources

- **Sentinel-2:** Microsoft Planetary Computer (Sentinel-2 Level-2A collection)
- **Study areas:** Mariupol, Kharkiv, and Bakhmut, Ukraine
- **Time span:** Q1 2021 – Q2 2024 (14 quarters, 41 scenes)
- **Damage labels:** Derived from conflict timeline (2021 = baseline, 2022–2024 = conflict)
- **Validation proxy:** UNOSAT-style period-level damage rate comparison

## Citation

If you use this project, please cite the proposal: "Automated Satellite Damage Assessment for Humanitarian Response in Ukraine"
