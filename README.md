# Satellite Intelligence Platform

This project applies big-data and machine-learning techniques to Sentinel-2 multispectral satellite imagery to automatically detect and classify conflict-related infrastructure damage across three Ukrainian cities — Mariupol, Kharkiv, and Bakhmut — from Q1 2021 through Q2 2024. Raw imagery is cleaned and cloud-masked in a preprocessing pipeline, spectral indices (NDVI, NDBI, BSI) and texture features are extracted at patch and scene level, and four classifiers (Logistic Regression, Random Forest, MLP, ResNet-50) are trained on a group-stratified split to avoid data leakage across city-year groups. The best model (MLP) achieves 93.4% test accuracy and 0.980 AUC-ROC. Results are surfaced through an interactive Streamlit dashboard with seven tabs covering model comparison, per-city metrics, spectral trends, prediction exploration, a satellite-imagery damage heatmap, and temporal animations.

---

## Group Members

| Name | Student ID |
|---|---|
| Josiah Alexis | 816040879 |
| Jonathan La Borde | 816041435 |
| Adrian Deo | 816042173 |
| Matthew Singh | 816035076 |

---

## Setup

**Python version:** 3.10 or higher (developed on Python 3.12).

**System dependencies:** None beyond Python itself. PyTorch and torchvision are required for ResNet-50 training; CPU-only builds are sufficient if no GPU is available.

**Create environment and install dependencies:**

```bash
git clone <repo-url>
cd Satellite-Intelligence-Platform

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> `requirements.txt` contains only the packages needed to run the Streamlit dashboard (what Streamlit Cloud installs). To also run the notebooks, API, and MLflow sections locally, install the additional dev dependencies:
> ```
> pip install jupyter torch torchvision tqdm planetary-computer pystac-client \
>             mlflow fastapi uvicorn requests pytest
> ```

---

## Running the Pipeline End to End

Run these commands in order from the **project root**. Each notebook must complete before the next one starts.

```bash
# 1. Activate the environment (if not already active)
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 2. Launch JupyterLab
jupyter lab
```

Inside JupyterLab, open and run each notebook in sequence:

| Step | Notebook | What it does |
|---|---|---|
| 1 | `satellite-damage-detection/notebooks/1-satellite_data_cleaning.ipynb` | Download Sentinel-2 imagery from Microsoft Planetary Computer, apply cloud masking, compute NDVI/NDBI/BSI, save `.npy` scenes |
| 2 | `satellite-damage-detection/notebooks/2-feature_engineering.ipynb` | Slice scenes into patches, assign damage labels, produce train/val/test splits |
| 2b *(optional)* | `satellite-damage-detection/notebooks/2b-rich_features.ipynb` | Enrich splits with GLCM texture and distribution features |
| 3 | `satellite-damage-detection/notebooks/3-model_training.ipynb` | Train LR, RF, MLP, and ResNet-50; save model files and test predictions |
| 4 | `satellite-damage-detection/notebooks/4-evaluation.ipynb` | Full evaluation, damage maps, per-city metrics, final summary JSON |

**Alternative — single merged notebook:**

```bash
# Open the combined pipeline (runs notebooks 1–4 in one file)
jupyter lab "satellite-damage-detection/notebooks/Satellite Intelligence Platform Merged.ipynb"
```

**Launch the Streamlit dashboard** (after running notebooks 1–4):

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**.

**Launch the FastAPI prediction service** (optional):

```bash
uvicorn api.app:app --reload --port 8000
```

Available at **http://localhost:8000** · docs at **http://localhost:8000/docs**.

**Run with Docker** (optional — no local Python environment required for the API):

```bash
docker compose up --build
```

---

## Reproducing the Main Results

The key results reported in the paper are produced by Notebook 4 (`4-evaluation.ipynb`) and depend on the model files and test split created by Notebooks 2 and 3. To reproduce:

1. Complete the full pipeline (steps 1–4 above).
2. Open `4-evaluation.ipynb` and run all cells.
3. Results are written to:
   - `satellite-damage-detection/results/metrics/final_summary.json` — overall test-set metrics
   - `satellite-damage-detection/results/metrics/model_comparison.csv` — per-model accuracy, F1, AUC
   - `satellite-damage-detection/results/metrics/per_city_metrics.csv` — per-city breakdown
   - `satellite-damage-detection/results/predictions/test_predictions.csv` — patch-level predictions

**Expected results (MLP, test set):**

| Metric | Value |
|---|---|
| Accuracy | 93.4% |
| Precision | 93.4% |
| Recall | 93.4% |
| F1 Score | 0.934 |
| AUC-ROC | 0.980 |

**Per-city (MLP):**

| City | F1 | AUC-ROC |
|---|---|---|
| Kharkiv | 1.000 | 1.000 |
| Bakhmut | 0.981 | 1.000 |
| Mariupol | 0.955 | 0.961 |

> Note: ResNet-50 was trained for fewer epochs due to compute constraints and scores lower than the classical models in this configuration.

---

## Data

**Source:** [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com) — Sentinel-2 Level-2A collection (`sentinel-2-l2a`). No account is required; data is accessed via the public STAC API in Notebook 1.

**Study areas and period:** Mariupol, Kharkiv, and Bakhmut, Ukraine · Q1 2021 – Q2 2024 · 41 scenes across 14 quarters.

**How to place data in `data/raw/`:**

Notebook 1 downloads and saves all scenes automatically. If you have pre-downloaded `.npy` files, place them directly in:

```
satellite-damage-detection/data/raw/
```

File naming convention: `S2_<YEAR>_<QUARTER>_<City>_clean.npy` for quarterly scenes (e.g. `S2_2022_Q2_Mariupol_clean.npy`) and `S2_<YEAR>_<City>_clean.npy` for annual composites.

Create the directory structure before running Notebook 1 (Windows PowerShell):

```powershell
New-Item -ItemType Directory -Force `
  satellite-damage-detection/data/raw, `
  satellite-damage-detection/data/processed, `
  satellite-damage-detection/data/patches, `
  satellite-damage-detection/data/splits, `
  satellite-damage-detection/models, `
  satellite-damage-detection/results/metrics, `
  satellite-damage-detection/results/predictions, `
  satellite-damage-detection/results/plots, `
  satellite-damage-detection/results/damage_maps
```

**Licensing:** Sentinel-2 imagery is provided by ESA/Copernicus under the [Copernicus Open Access Data Policy](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice). Data accessed through Microsoft Planetary Computer is subject to the [Planetary Computer Terms of Use](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#License). Free for non-commercial and research use with attribution.

---

## Demo

[![Demo Video](https://img.youtube.com/vi/gxH328IiMFg/0.jpg)](https://www.youtube.com/watch?v=gxH328IiMFg)

[Live Streamlit App]([https://big-data-analysis-group-satellite-intelligence-dashboard-sdi6ll.streamlit.app/](https://big-data-analysis-group-satellite-intelligence-platf-app-48n5nq.streamlit.app/))

The dashboard has seven tabs:

| Tab | Content |
|---|---|
| Overview | Key metrics, dataset split, project summary |
| Model Comparison | Accuracy / F1 / AUC bar chart, interactive model filter |
| Per-City Analysis | Per-city metrics, scene counts, damaged-scene breakdown |
| Spectral Trends | NDVI / NDBI / BSI over time with conflict-onset marker |
| Predictions | Threshold slider, live confusion matrix, probability histogram, scatter plots |
| Damage Heatmap | Satellite RGB background + 64×64 px damage overlay, yearly and quarterly modes |
| City Animations | Animated GIF cycling 2021–2024 for each city |

---

## Project Structure

```
Satellite-Intelligence-Platform/
├── satellite-damage-detection/
│   ├── notebooks/
│   │   ├── 1-satellite_data_cleaning.ipynb
│   │   ├── 2-feature_engineering.ipynb
│   │   ├── 2b-rich_features.ipynb          # optional texture features
│   │   ├── 3-model_training.ipynb
│   │   ├── 4-evaluation.ipynb
│   │   └── Satellite Intelligence Platform Merged.ipynb
│   ├── src/                # Python source modules
│   ├── data/               # raw imagery, processed CSVs, patches, splits
│   ├── models/             # trained model files (.pkl / .pth)
│   └── results/            # metrics, predictions, plots, damage maps
├── api/
│   ├── app.py              # FastAPI prediction service
│   └── requirements.txt    # API-specific dependencies
├── tests/
│   ├── test_demo.py        # End-to-end demo/test (FastAPI + Docker + MLflow)
│   └── test_api.py         # API unit tests (11 tests, pytest)
├── docs/
│   ├── Satallite-Intelligience-Plateform-COMP3610-Final-Report.pdf
│   └── Satellite_Intelligence_Platform_COMP3610-Presentation.pdf
├── app.py                  # Streamlit dashboard (7 tabs)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt        # Full environment freeze
```

---

## AI Tools Used

GitHub Copilot and Claude (Anthropic) were used during development for debugging, and refactoring across the notebooks, API, and Streamlit dashboard. Claude was also used to assist with README formatting and to explain concepts related to spectral indices and satellite imagery processing. All generated code was reviewed, tested, and validated by the team.
