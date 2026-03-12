# Satellite Damage Detection - Ukraine Conflict Analysis

Automated damage assessment using Sentinel-2 satellite imagery to detect and classify infrastructure damage across Ukrainian conflict zones (Mariupol, Kharkiv, Bakhmut).

## Project Structure

```
satellite-damage-detection/
├── notebooks/          # Jupyter notebooks (analysis & experiments)
├── src/               # Python source code
├── data/              # Data directory (raw, processed, splits)
├── models/            # Trained models & checkpoints
├── results/           # Outputs, metrics, visualizations
└── scripts/           # Standalone executable scripts
```

## Setup

1. **Clone and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up data directories:**
   ```bash
   mkdir -p data/{raw,processed,splits}
   mkdir -p models/checkpoints
   mkdir -p results/{metrics,predictions,plots}
   ```

## Workflow

1. **Data Cleaning** (`notebooks/01_data_cleaning.ipynb`)
   - Download Sentinel-2 Level-2A from Microsoft Planetary Computer
   - Filter 2021-2024 data for study areas
   - Apply cloud masking

2. **Feature Engineering** (`notebooks/02_feature_engineering.ipynb`)
   - Compute NDVI, NDBI, BSI indices
   - Generate temporal features
   - Create image patches

3. **Model Training** (`notebooks/03_model_training.ipynb`)
   - Fine-tune ResNet-50 on EuroSAT/UC Merced
   - Train damage classifier

4. **Evaluation** (`notebooks/04_evaluation.ipynb`)
   - Validate against UNOSAT assessments
   - Generate damage maps

## Key Files

- `src/data.py` - Load & preprocess Sentinel-2 imagery
- `src/features.py` - Spectral index calculations
- `src/model.py` - ResNet-50 architecture
- `src/train.py` - Training loop
- `src/utils.py` - Helper functions

## Usage

**Train a model:**
```bash
python scripts/train.py --epochs 50 --batch-size 32
```

**Evaluate:**
```bash
python scripts/evaluate.py --model-path models/checkpoints/best_model.pth
```

## Data Sources

- **Sentinel-2:** Microsoft Planetary Computer
- **Training:** EuroSAT, UC Merced
- **Validation:** UNOSAT damage assessments

## Citation

If you use this project, please cite the proposal: "Automated Satellite Damage Assessment for Humanitarian Response in Ukraine"
