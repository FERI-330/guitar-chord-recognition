# Guitar Chord Recognition

This repository now uses a manifest-driven image classification pipeline for guitar chord recognition.
The source of truth is `data/split_manifest.csv`, generated from `data/all/` by a stratified 70/15/15 split.

## Current workflow

1. **EDA** (`notebooks/01_EDA.ipynb`): Explore the data, check for issues, and gather statistics.
2. **Manifest** (`notebooks/02_split_manifest.ipynb`): Generate stratified 70/15/15 split from `data/all/`.
3. **Preprocessing** (`notebooks/03_preprocessing.ipynb`): Set up PyTorch DataLoaders, augmentations, and class weights.
4. **Data Leakage Check** (`notebooks/04_data_leakage_check.ipynb`): Verify no duplicate images between train/val/test using MD5 and pHash.

## Progress

- Edge-first fretboard homography implemented in `notebooks/03_feature_pipeline.ipynb` (detect_neck_lines + fit_corners_from_lines). The pipeline now prefers edges-derived corners and falls back to bbox-based homography.
- Batch feature extraction completed: 297 images → features of dimension 139 (train 207, val 45, test 45). Sanity checks show no NaN/Inf in feature matrices.
- Known issue: a non-trivial subset of images produce faulty/low-confidence fretboard detections (low `bund_det_rate` or `H_valid==0`). Diagnostics and triage are pending — see `JOURNAL.md` for details and next steps.
- Next: export CSV of weak detections, triage failure modes, tune Hough/fallback parameters, and re-run batch extraction.
5. **Model Training** – Choose a notebook based on your architecture:
   - `notebooks/04a_baseline_ml.ipynb` – Scikit-learn baseline models (baseline_ml)
   - `notebooks/04b_mobile_cnn.ipynb` – MobileNet v3 (small & large variants)
   - `notebooks/04c_efficientnet_b0.ipynb` – EfficientNet-B0 (two training phases: phA, phB)
   - `notebooks/04d_advanced_cnn.ipynb` – Advanced CNN architectures (two phases: phA, phB)

## Repository layout

```
.
├── data/
│   ├── all/                    # Source images (A, B, C, D, E, F, G, No hand)
│   ├── training/               # Reference folder (legacy)
│   ├── test/                   # Reference folder (legacy)
│   └── split_manifest.csv      # Split source of truth (70/15/15)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_split_manifest.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_data_leakage_check.ipynb
│   ├── 04a_baseline_ml.ipynb
│   ├── 04b_mobile_cnn.ipynb
│   ├── 04c_efficientnet_b0.ipynb
│   └── 04d_advanced_cnn.ipynb
├── output/                     # All notebook outputs (figures, results)
├── checkpoints/                # Trained model weights
├── src/
│   ├── models.py
│   └── train.py
├── environment.yaml
├── requirements.txt
├── JOURNAL.md
└── README.md
```

## Environment

The working environment is Python 3.10 with CUDA 12.4 support on an NVIDIA T500 GPU.
PyTorch is intentionally installed via `pip` with cu124 wheels.

Do not install PyTorch from conda channels in this project. Use the documented pip wheels instead:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Recommended setup for the remaining scientific stack:

```bash
conda activate guitar-chord
conda install -y -c conda-forge numpy pandas matplotlib seaborn scikit-learn pillow tqdm pygments jupyter ipykernel
python -m ipykernel install --user --name guitar-chord --display-name "guitar-chord"
```

## Data summary

- Total images: 297
- Classes: `A`, `B`, `C`, `D`, `E`, `F`, `G`, `No hand`
- Split sizes: train 207, val 45, test 45

The manifest keeps the raw folder structure untouched and records the split assignment per image.

## Preprocessing decisions

- Image size: 224 x 224
- Normalization: ImageNet mean/std for transfer learning
- Train augmentations: horizontal flip, color jitter, random rotation
- Validation and test: resize + center crop + normalize
- Batch size: 16
- Class weighting: inverse frequency, with `G` and `No hand` weighted higher

## Training entry points

### Notebooks (Interactive)

Each notebook handles a complete training pipeline with results saved to `output/<notebook_name>/`:

- **`notebooks/04a_baseline_ml.ipynb`** – Scikit-learn baseline (LogisticRegression, SVM, RandomForest)
- **`notebooks/04b_mobile_cnn.ipynb`** – MobileNet v3 small and large
- **`notebooks/04c_efficientnet_b0.ipynb`** – EfficientNet-B0 with class-weighted loss
- **`notebooks/04d_advanced_cnn.ipynb`** – Custom advanced CNN architecture

Each notebook saves:
- **Results**: `output/<name>/results.csv` with metrics (accuracy, precision, recall, F1)
- **Checkpoints**: Best model weights to `checkpoints/best_<model>_ph{A,B}.pth`
- **Figures**: Training curves, confusion matrices, class-wise metrics

### Script

`src/train.py` provides a reproducible training pipeline:

```bash
python -m src.train
```

Or call it with explicit arguments:

```python
from src.train import main

main(
    manifest_path="data/split_manifest.csv",
    batch_size=16,
    img_size=224,
    epochs=50,
    model_name="efficientnet_b0",
)
```

## Model baselines

Multiple architectures are trained and compared:

| Architecture | Checkpoint(s) | Notes |
|--------------|--------------|-------|
| **Baseline ML** | `baseline_ml/` | Scikit-learn models (baseline for comparison) |
| **MobileNet v3** | `best_MobSmall_phA.pth`, `best_MobSmall_phB.pth`, `best_MobLarge_phA.pth`, `best_MobLarge_phB.pth` | Lightweight CNN (small: 2.5M params, large: 5.4M params) |
| **ShuffleNet v2** | `shufflenet_v2_x1_0.pth`, `best_shuffle_phA.pth`, `best_shuffle_phB.pth` | Efficient architecture optimized for mobile |
| **EfficientNet-B0** | `best_EfficientNet_phA.pth`, `best_EfficientNet_phB.pth` | Balanced accuracy/efficiency baseline |
| **Advanced CNN** | `best_AdvancedCNN_phA.pth`, `best_AdvancedCNN_phB.pth` | Custom deeper architecture for higher capacity |

**Training Phases:**
- **Phase A (phA):** Initial training on full dataset
- **Phase B (phB):** Fine-tuning or transfer learning continuation

## Outputs

- **Notebook outputs**: Figures, metrics, and results saved under `output/<notebook_name>/`
- **Model checkpoints**: Best weights from each architecture/phase saved to `checkpoints/best_<model>_ph{A,B}.pth`
- **Development log**: See `JOURNAL.md` for implementation decisions, hyperparameters, and debugging notes

## Notes

- **Data source**: `data/all/` (297 images across 8 classes)
- **Split source of truth**: `data/split_manifest.csv` (generated by `02_split_manifest.ipynb`)
- **Legacy folders**: `data/training/` and `data/test/` remain for reference only—they are not used by the new manifest-based pipeline
- **Reproducibility**: All notebooks use `random_state=42` for deterministic splits and initialization
- **PyTorch setup**: Install PyTorch via pip with CUDA support to avoid conda libtorch conflicts:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```
- **Data leakage**: Run `04_data_leakage_check.ipynb` to verify train/val/test independence using MD5 and pHash
- **Detailed log**: See `JOURNAL.md` for development decisions, hyperparameter choices, and debugging notes