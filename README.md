# Guitar Chord Recognition

This repository now uses a manifest-driven image classification pipeline for guitar chord recognition.
The source of truth is `data/split_manifest.csv`, generated from `data/all/` by a stratified 70/15/15 split.

## Current workflow

1. Build the manifest in `notebooks/02_split_manifest.ipynb`.
2. Preprocess and inspect batches in `notebooks/03_preprocessing.ipynb`.
3. Train the EfficientNet-B0 baseline in `notebooks/04_model.ipynb` or via `src/train.py`.

## Repository layout

```
.
├── data/
│   ├── all/
│   ├── training/
│   ├── test/
│   └── split_manifest.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_split_manifest.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_model.ipynb
├── output/
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

### Notebook

Open `notebooks/04_model.ipynb` for the interactive training workflow.

### Script

`src/train.py` provides a small reproducible pipeline:

```bash
python -m src.train
```

Or call it with explicit arguments from Python:

```python
from src.train import main

main(
    manifest_path="data/split_manifest.csv",
    batch_size=16,
    img_size=224,
    epochs=1,
)
```

## Model baseline

The first baseline is EfficientNet-B0 from `torchvision`, with the classifier replaced for 8 classes.
The training loop uses class-weighted cross entropy and saves the best checkpoint to `checkpoints/best_model.pth`.

## Outputs

- Notebook figures are written under `output/<notebook_name>/`
- The training checkpoint is written to `checkpoints/best_model.pth`
- The journal in `JOURNAL.md` records major implementation and debugging decisions

## Notes

- `data/training/` and `data/test/` remain as reference folders only.
- `data/split_manifest.csv` is the only split source used by the new notebooks.
- If you hit CUDA or import issues, check `JOURNAL.md` for the pip-based PyTorch setup that avoids the conda `libtorch` conflict.