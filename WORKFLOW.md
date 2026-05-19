# Pipeline Workflow Dokumentáció

## 1. Folyamatábra: Nyers képtől a predikciókig

```
RAW KÉPEK
data/all/<chord>/*.jpg
        │
        ▼
[02_split_manifest]  ──→  data/split_manifest.csv
 (egyszer fut, ha                │
  új kép kerül be)               │
        │                        │
        ▼                        ▼
[03_pipeline]  ←── src/fretboard.py, hand_landmark.py,
 ROI-kivágás,          geometry.py, constants.py,
 landmark-detektálás   config.py, preprocess.py
        │
        ▼
[features.py:extract_features_batch()]
        │
        ▼
data/features/features_v14.npz  ←── CHECKPOINT #1
        │
        ├──────────────────────────────┐
        ▼                              ▼
[05a_baseline_ml]              [05b_cnn_finetune]
 SVM tanítás                    MobileNetV3 fine-tuning
        │                              │
        ▼                              ▼
checkpoints/                   checkpoints/              ←── CHECKPOINT #2
  best_ml_model.pkl               best_mobilenet_v3_large_phB.pth
        │                              │
        └──────────────┬───────────────┘
                       ▼
              [06_evaluation]
               Végső metrikák
                       │
                       ▼
         output/06_evaluation/final_results.json  ←── CHECKPOINT #3
```

---

## 2. Pipeline Függőségi Térkép

**Ha ezt módosítod → ezeket kell újra futtatni:**

| Módosított fájl/adat | 02 | 03 | 05a | 05b | 06 |
|---|---|---|---|---|---|
| `src/preprocess.py` (CLAHE, blur) | ✗ | ✓ | ✓ | ✓ | ✓ |
| `src/fretboard.py` (ROI logika) | ✗ | ✓ | ✓ | ✓ | ✓ |
| `src/features.py` (feature vektor) | ✗ | ✓ | ✓ | ✓ | ✓ |
| `src/hand_landmark.py` | ✗ | ✓ | ✓ | ✓ | ✓ |
| `src/config.py` (csak viz params) | ✗ | ✗ | ✗ | ✗ | ✗ |
| `src/models.py` / `train.py` | ✗ | ✗ | ✓ | ✓ | ✓ |
| Új képek kerülnek `data/all/`-ba | ✓ | ✓ | ✓ | ✓ | ✓ |
| Csak vizualizáció változik (06) | ✗ | ✗ | ✗ | ✗ | ✓ |

**Aranyszabály:** a módosított fájltól lefelé kell futtatni. A fájlok hierarchiája:
`preprocess/fretboard/features` → `features.npz` → `05a/05b` → `06`

---

## 3. Kernel Restart vs. Re-run

**Miért kötelező Kernel Restart `src/` módosítás után?**

Python `import` az első `import src.preprocess` után **cache-eli** a modult a `sys.modules`-ban.
Ha utána módosítod a `.py` fájlt, a notebook **továbbra is a régi verziót látja** — újraindítás nélkül.

```
1. szerkeszted src/preprocess.py
2. notebook cellát lefuttatod
3. Python NEM tölti újra → régi kód fut
```

**Két megoldás:**

```python
# Gyors megoldás egy cellánál (nem tökéletes mély függőségeknél):
import importlib, src.preprocess
importlib.reload(src.preprocess)

# Megbízható megoldás: Kernel → Restart & Run All
```

**Mikor elég csak újrafuttatni (restart nélkül)?**
- Ha csak egy **adat-fájl** változott (`features_v14.npz` frissült) és a betöltő cella még nem futott le — egyszerűen futtasd a betöltő cellát újra.
- Ha a változás **egy másik notebookban** volt, és az aktuális csak betölt fájlokat (nem importál `src/`-t).

**Összefoglalva:**

| Változás típusa | Teendő |
|---|---|
| `src/*.py` módosult | **Kernel Restart + Run All** |
| `.npz` / `.pkl` / `.pth` frissült | Csak a betöltő cellától futtasd újra |
| Notebook vizualizációs cella | Csak az adott cellát futtasd |
| `config.py` konstans változott | Kernel Restart (konstansok cache-elve) |

---

## 4. Automatizálási javaslat: `main.py`

A projekt már rendelkezik `src/inference.py`-val és minden modullal — egy `main.py` minimális plusz munkával megírható:

```python
# main.py (javasolt struktúra)
from src.fretboard import run_v14_pipeline
from src.features import assemble_feature_vector
from src.inference import predict   # ha létezik, vagy models.py-ból

def predict_chord(image_path: str) -> str:
    result = run_v14_pipeline(image_path)
    feat = assemble_feature_vector(result)
    return predict(feat)            # SVM vagy CNN

if __name__ == "__main__":
    import sys
    print(predict_chord(sys.argv[1]))
```

Futtatás: `python main.py data/test/kep.jpg`

Ez teljes pipeline futtatáshoz elegendő — a notebookok tanítás/kiértékelésre valók, az inference már szkriptelhető.

---

## 5. "Plug-and-Play" szabályok

| Kérdés | Válasz |
|---|---|
| Csak vizualizáción változtatok (06)? | **Nem kell** újratanítani semmit. Töltsd be a kész `.npz`/`.pkl`/`.pth` fájlokat. |
| Új preprocessing, de ugyanazok a képek? | 03-tól kell újra (features.npz regenerálás), majd 05a+05b+06 |
| Csak SVM hyperparamétert hangolok? | Csak 05a, majd 06 |
| Csak CNN-t finom-hangolom? | Csak 05b, majd 06 |
| Teljesen új adathalmaz? | 02-től minden |

A `features_v14.npz` és a `checkpoints/` a két kulcs-checkpoint — ha ezek megvannak, 06 bármikor újrafut.
