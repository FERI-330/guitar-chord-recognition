# Guitar Chord Recognition

Statikus képekből gitár akkord osztályozása 8 osztályra (A, B, C, D, E, F, G, No hand),
297 képből álló adathalmazon.

## Eredmények

| Modell | Val Acc | Test Acc | Test F1 |
|---|---|---|---|
| **MobileNetV3-Large** (Phase A+B) | 97.8% | **97.8%** | **0.971** |
| SVM (Group B, 42 dim) | 95.6% | 91.1% | 0.907 |

A CNN 6.7%-kal jobb a teszt halmazon → MobileNetV3-Large az ajánlott modell.

---

## Architektúra áttekintés

A projekt kétszintű megközelítést alkalmaz:

1. **V14 Pipeline** – OpenCV + MediaPipe alapú fogólap- és ujjhegy-detektálás, 56 dimenziós feature vektor előállítása
2. **CNN Fine-tuning** – MobileNetV3-Large két-fázisú transfer learning nyers képeken

```
Kép → run_v14_pipeline → 56-dim feature → SVM (91.1%)
    ↘ MobileNetV3-Large fine-tune    → CNN (97.8%)
```

---

## Telepítés

```bash
# Conda környezet létrehozása
conda create -n guitar-chord python=3.10
conda activate guitar-chord

# PyTorch CUDA-val (pip, ne conda)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Tudományos stack
conda install -y -c conda-forge numpy pandas matplotlib seaborn scikit-learn \
    pillow tqdm jupyter ipykernel xgboost

# MediaPipe
pip install mediapipe

# Jupyter kernel regisztrálás
python -m ipykernel install --user --name guitar-chord --display-name "guitar-chord"
```

> **MediaPipe model**: `models/hand_landmarker.task` szükséges a pipeline futtatásához.
> Töltsd le a [MediaPipe Models](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) oldalról.

---

## Adatstruktúra

```
data/
├── all/                    # Forrás képek (297 db, 8 osztály)
│   ├── A/                  # A akkord képek
│   ├── B/
│   ├── C/
│   ├── D/
│   ├── E/
│   ├── F/
│   ├── G/
│   └── No hand/
├── split_manifest.csv      # Stratified 70/15/15 split forrás
└── features/
    └── features_v14.npz    # 297×56 feature mátrix (V14 pipeline output)
```

**Split:** 207 train / 45 val / 45 test (stratified, random_seed=42)

---

## `src/` modul struktúra

Függőségi sorrend (alulról felfelé):

```
config → constants → geometry → hand_landmark → fretboard
                                              ↓
                              features → dataset → models → train → viz
```

| Modul | Felelősség |
|---|---|
| `config.py` | `CFG` és `PATHS` dict – egyetlen igazságforrás minden konstanshoz |
| `constants.py` | `CFG`-ből számított domain-specifikus tömbök (fret-pozíciók, landmark indexek) |
| `geometry.py` | OpenCV fretboard geometria: Canny, Hough, trapézoid detektálás, perspective warp |
| `hand_landmark.py` | MediaPipe kézdetektálás, landmark projekció, ujjmaszk |
| `fretboard.py` | `run_v14_pipeline` orchestrátor (15 lépéses pipeline) |
| `features.py` | 56-dim feature vektor összeállítás, batch extrakció, NPZ mentés/betöltés |
| `dataset.py` | PyTorch `Dataset`, `DataLoader`, augmentációk, class weight számítás |
| `models.py` | CNN builder dispatcher (`build_model`), freeze/unfreeze segédletek |
| `train.py` | `EarlyStopping`, `train_one_epoch`, `evaluate`, `train_two_phase` |
| `viz.py` | Pipeline és training vizualizáció (pipeline overlay, training görbék, scatter plotok) |

---

## V14 Pipeline részletei

A `run_v14_pipeline` 15 lépéses pipeline:

1. **Canny** éldetektálás (ujjmaszk alapú szűréssel)
2. **Hough** vonaldetektálás
3. **Nyakszög** meghatározás (landmark anchor fallback-kel)
4. **Vonalak szétválasztása** (hossz szerint)
5. **Külső élek** keresése (fogólap bal/jobb széle)
6. **Trapézoid** sarokpontok meghatározása
7. **Trapézoid validálás** (aspect ≥ 4.0, area_frac ∈ [0.010, 0.50], edge_angle_diff ≤ 15°)
8. **Perspective warp** → 600×80 px kanonikus tér
9. **Nut detektálás** + anchor override
10. **Nut-trim + re-warp**
11. **Bundvonalak** detektálása a kanonikus képen
12. **Ujjpár-szuppresszió** (dupla éldetek eltávolítása)
13. **17.817-es bund-szabály illesztés** (RANSAC-szerű)
14. **Ujjhegy-vetítés** a kanonikus térbe
15. **Feature vektor** összeállítás (56 dim)

**Pipeline ok-rate:** 248/297 = 83.5%

### Feature vektor (56 dim)

| Csoport | Dim | Tartalom |
|---|---|---|
| B | 42 | Wrist-normalized landmark x,y (21 pont × 2) |
| D | 2 | Detektálási flagek (hand_detected, fretboard_detected) |
| F | 2 | Nyakszög cos/sin |
| G | 5 | Bund-index per ujj (normalizált, 0=nem detektált) |
| H | 5 | Húr-pozíció per ujj (normalizált, 0=nem detektált) |

> **ok=False policy:** Group B megmarad (ha kéz látható), G/H = 0, D = (1, 0), F = 0.

---

## Notebookok

| Notebook | Cél |
|---|---|
| `01_EDA.ipynb` | Dataset profilozás, osztályeloszlás, képminőség vizsgálat |
| `02_split_manifest.ipynb` | Stratified 70/15/15 split generálása |
| `03_pipeline.ipynb` | Batch V14 pipeline futtatás, `features_v14.npz` mentés, failure triage |
| `04_feature_analysis.ipynb` | PCA, t-SNE, korreláció, group ablation |
| `05a_baseline_ml.ipynb` | SVM / RF / XGBoost baseline a feature vektoron |
| `05b_cnn_finetune.ipynb` | MobileNetV3 két-fázisú fine-tuning |
| `06_evaluation.ipynb` | **Egyetlen** hely ahol a test set betöltődik – végső összehasonlítás |

> **Test set védelem:** a test splitet kizárólag `06_evaluation.ipynb` tölti be.

---

## Inference – egy kép kiértékelése

### CNN (ajánlott, 97.8% teszt acc)

```python
import torch
from src.config import CFG, PATHS
from src.models import build_model
from src.train import load_checkpoint
from src.dataset import get_transforms
from PIL import Image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'No hand']

model = build_model('mobilenet_v3_large', num_classes=8).to(DEVICE)
model = load_checkpoint(model, PATHS['checkpoint_dir'] / 'best_mobilenet_v3_large_phB.pth', DEVICE)
model.eval()

transform = get_transforms('val')
img = transform(Image.open('data/all/A/kep.jpg').convert('RGB')).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(img)
    pred = logits.argmax(1).item()

print(f'Predikció: {CLASSES[pred]}')
```

### SVM (GPU nélkül, 91.1% teszt acc)

```python
import pickle
import numpy as np
from src.fretboard import run_v14_pipeline
from src.features import assemble_feature_vector

with open('checkpoints/best_ml_model.pkl', 'rb') as f:
    ml_data = pickle.load(f)
svm = ml_data['model']

result = run_v14_pipeline({'path': 'data/all/A/kep.jpg', 'class': 'A'})
feat = assemble_feature_vector(result)
B_cols = list(range(42))

pred_label = svm.predict(feat[B_cols].reshape(1, -1))[0]
print(f'Predikció: {pred_label}')
```

---

## Vizualizáció (src/viz.py)

```python
from src.viz import draw_pipeline_result, plot_training_history

# Pipeline eredmény megjelenítése
result = run_v14_pipeline({'path': 'data/all/A/kep.jpg', 'class': 'A'})
fig = draw_pipeline_result(result)
fig.savefig('output/pipeline_demo.png', dpi=130)

# Training görbék
import json
with open('output/05b_cnn_finetune/best_cnn_meta.json') as f:
    meta = json.load(f)
fig = plot_training_history(meta['history'], title='MobileNetV3-Large')
```

---

## Checkpointok

| Fájl | Leírás |
|---|---|
| `checkpoints/best_mobilenet_v3_large_phB.pth` | **Legjobb modell** – MobileNetV3-Large Phase B |
| `checkpoints/best_mobilenet_v3_large_phA.pth` | MobileNetV3-Large Phase A |
| `checkpoints/best_mobilenet_v3_small_phB.pth` | MobileNetV3-Small Phase B (91.1% val) |
| `checkpoints/best_ml_model.pkl` | Legjobb ML modell – SVM_B (sklearn pickle) |

---

## Környezet

- Python 3.10, CUDA 12.4, NVIDIA T500 GPU
- PyTorch: pip wheels (ne conda channels – libtorch konfliktus)
- Reprodukálhatóság: `random_seed=42` minden notebookban és `src/config.py`-ban

---

## Fejlesztési napló

Részletes fejlesztési döntések, hibajavítások és kísérleti eredmények: [JOURNAL.md](JOURNAL.md)

Főbb mérföldkövek:
- V14 pipeline: `validate_trapezoid` `hand_inside` ellenőrzés eltávolítva (49% false-reject → 16.5%)
- Feature vektor: 139 dim → 56 dim (Group B + D + F + G + H)
- CNN győz: +6.7% teszt acc a legjobb SVM felett → CNN ajánlott
