# Satellite Damage Detection - Ukraine Conflict Analysis

Automated damage assessment using Sentinel-2 satellite imagery to detect and classify infrastructure damage across Ukrainian conflict zones (Mariupol, Kharkiv, Bakhmut).

## Project Structure

```
satellite-damage-detection/
├── notebooks/
│   ├── 1-satellite_data_cleaning.ipynb       # Data download, cloud masking, spectral indices
│   ├── 2-feature_engineering.ipynb            # Patch/scene dataset preparation and splits
│   ├── 2b-rich_features.ipynb                 # Optional: GLCM texture & distribution features
│   ├── 3-model_training.ipynb                 # Train LR, RF, MLP, and ResNet-50
│   ├── 4-evaluation.ipynb                     # Full evaluation, damage maps, reports
│   └── Satellite Intelligence Platform Merged.ipynb  # Combined single-file pipeline
├── src/               # Python source modules
├── data/              # Data directory (raw, processed, patches, splits)
├── models/            # Trained model files and preprocessing artifacts
├── results/           # Outputs: metrics, predictions, plots, damage maps
└── scripts/           # Standalone executable scripts
```

## Setup

1. **Clone and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up data directories:**
   ```bash
   mkdir -p data/{raw,processed,patches,splits}
   mkdir -p models
   mkdir -p results/{metrics,predictions,plots,damage_maps}
   ```

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
