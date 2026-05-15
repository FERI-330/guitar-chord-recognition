# 📜 Projekt Fejlesztési Napló – Gitár Akkord Felismerő

**Projekt:** Gitár akkord felismerő szoftver gépi látással  
**Szerző:** Magda Ferenc (U5O0BB)  
**Dokumentum típusa:** Retrospektív fejlesztési napló – minden bejegyzés megmarad, semmi nem törlődik.

---

## 🗓️ 2026-05-12 – EDA fázis (`01_EDA.ipynb`)

### Elvégzett műveletek

- Adathalmaz teljes körű feltérképezése mindhárom split mentén (`all/`, `training/`, `test/`)
- 10 szekciós EDA pipeline implementálása
- Vizualizációk és statisztikák generálása
- Döntési táblázat összeállítása a preprocessing fázishoz

### Felmerült kihívások

#### Adatstruktúra komplexitása
- 3 különálló mapparendszer párhuzamos kezelése (`all/`, `training/`, `test/`)
- Egyenletes inventoryzálás minden splitre
- Pivot táblázatok összeépítése a három dataset összevetéséhez

#### Képminőség- és integritás-ellenőrzés
- Sérült képek detektálása (`PIL.verify()` módszerrel)
- MD5 hash-alapú duplikátum-keresés train–test között (leakage detektálás)
- Megjegyzés: ha lett volna duplikátum, azt korrekciós logika nélkül hibaként azonosítja

#### RGB csatorna-statisztikák
- Pixelintenzitás-eloszlások kinyerése (mintaalapú: ~80 kép)
- Döntési dilemma: ImageNet default értékek vs. saját mean/std kalkuláció
- Végeredmény: saját normalizálási értékek meghatározása szükséges (R≈0.51, G≈0.44, B≈0.42)

#### Osztályeloszlás-analízis
- Imbalance ratio vizsgálata (automatikus figyelmeztetés 2×, 3× tényezőnél)
- Class weighting szükségessége a kiegyensúlyozatlan adatokhoz

#### Fájlforrás-heterogenitás
- Kamera által készített képek (`IMG_` prefix) vs. app által generált (timestamp)
- Különböző forrásból származó képek potenciális minőség-eltérései

#### Képméret-sokszínűség
- 224×224-re normalizálás szükséges, de az eredeti képek széles tartományban mozognak
- Aspect ratio megtartás vs. egyszerű resize dilemmája

### Státusz
✅ Adathalmaz profiling, minőség-ellenőrzés, döntési táblázat (10 pont) elkészítve

---

## 🗓️ 2026-05-12 – Split Manifest fázis (`02_split_manifest.ipynb`)

### Elvégzett műveletek

- Stratified 70/15/15 felosztás generálása az `all/` mappából
- `data/split_manifest.csv` létrehozása mint egyetlen, központi igazságforrás
- Split-konzisztencia ellenőrzése pivot táblával és vizualizációval

### Felmerült kihívások

#### Stratified split logika
- Kétlépéses `StratifiedShuffleSplit` implementálása szükséges:
  1. Először train (70%) vs. temp (30%) szétválasztás
  2. Majd a temp felezése → val (15%) + test (15%)
- Azért szükséges a kétlépés, mert a 15/15 arányok garantálásához előbb a 30%-ot kell kiemelni, majd azt felezni

#### Reprodukálhatóság
- `random_state=42` rögzítése szükséges a teljes pipeline determinisztikus maradásához
- Seed értékek konzisztens kezelése mindkét split-generátor között

#### Adatintegritás
- Semmilyen kép nem veszhet el a split során
- Oszlopok sorrendjének standardizálása: `split, class, filename, path, size_kb`
- CSV mentés és beolvasás során encoding-problémák elkerülése

#### Osztály-reprezentáció
- G és "No hand" osztályok biztosítása minden splitben
- Alulreprezentált osztályok (E, G, No hand) megfelelő kezelése mindhárom splitben

#### Validáció hiánya az eredeti adathalmazban
- Az eredeti `training/` mappa nem tartalmaz val/test szeparációt
- Megoldás: az új manifest-alapú rendszer kiváltja a régi struktúrát

### Eredmények

| Split | Képszám | Arány |
|-------|---------|-------|
| Train | 207 | 69.7% |
| Val | 45 | 15.2% |
| Test | 45 | 15.2% |

| Osztály | Train | Val | Test |
|---------|-------|-----|------|
| A | 23 | 5 | 5 |
| B | 39 | 8 | 9 |
| C | 33 | 7 | 7 |
| D | 31 | 6 | 7 |
| E | 22 | 5 | 4 |
| F | 34 | 8 | 7 |
| G | 15 | 3 | 4 |
| No hand | 10 | 3 | 2 |

### Státusz
✅ Reprodukálható stratified split, manifest CSV generálva, preprocessing fázis előkészítve

---

## 🔜 Nyitott döntési pontok – Preprocessing fázis (`03_preprocessing.ipynb`)

| Szempont | Döntés szükséges |
|----------|-----------------|
| Célméret | 224×224 vagy más? |
| Normalizálás | Saját mean/std (R≈0.51, G≈0.44, B≈0.42) vs. ImageNet default |
| Augmentáció típusa | RandomHorizontalFlip, ColorJitter, Rotation(±15°) mértéke |
| Class weighting | Melyik osztályok kapnak magasabb súlyt |
| MediaPipe hand-keypoints | Implementáció szükséges-e az előzetes feldolgozásban |

---

*Ez a dokumentum folyamatosan bővül. Semmi nem törlődik – minden döntés és fázis visszakereshető.*

---

## 🗓️ 2026-05-15 – Naplófrissítés: README és rövid összefoglaló

- Röviden: él-alapú (edge-first) fogólap-detektálás implementálva (`notebooks/03_feature_pipeline.ipynb`), batch feature-extrahálás lefutott (297 kép, feature-dim=139), de sok hibás detektálás maradt. README frissítve a rövid státusszal.
- Következő lépés: exportálom a gyenge detekciójú képek listáját (CSV), majd triage + Hough/fallback finomhangolás. Lásd TODO-lista.


---

## 🗓️ 2026-05-15 – Folyamatban (geometriai modell és detektálás)

### Rövid státusz

- Elvégeztem az él-alapú (edge-first) fogólap-modellezés implementálását a `03_feature_pipeline.ipynb` notebookban: van teljes képes vonaldetektor, vonalakból sarokpont-felépítés és a homográfia-illesztés preferálva az eredeti bbox-fallback mellett.
- A notebook lokális sanity check és a teljes batch feature-extrahálás lefutott: kinyert feature-ek shape-je változatlanul 139 dim lett, nincs NaN/Inf.

### Miért NEM kész még

- A feldolgozott képek között jelentős számú hibás vagy pontatlan `fretboard` detektálás található; ezek rontják az automatikus feature-extrahálás minőségét és a downstream modell-tréninget.
- Konkrétan: sok kép esetén a bundvonalak rosszul detektálódnak vagy a homográfia túl nagy hibát tartalmaz (alacsony `bund_det_rate` vagy `H_valid==0`).

### Következő lépések (röviden)

1. Diagnosztika futtatása: exportálom a gyenge detekciójú képek listáját (alacsony `bund_det_rate`, `H_valid==0`).
2. Triage: csoportosítás ok szerint (pl. részben vágott fogólap, tükröződés, képtorzulás, rossz Hough-paraméterek). 
3. Finomhangolás: szigorítás/ellazítás Hough küszöbök, per-sample confidence küszöb bevezetése, és szükség esetén fallback-szabályok javítása.
4. Újra futtatás és validáció, majd Journal lezárása.

### Megjegyzés
- A teendőlista frissítve: diagnosztika folyamatban (lásd TODO-lista). Nem zárom le a Journal-t addig, amíg a hibák triage/korrekciója nem történik meg.


---

## 🗓️ 2026-05-12 – Preprocessing fázis (`03_preprocessing.ipynb`)

### Elvégzett műveletek

- `GuitarChordDataset(Dataset)` PyTorch osztály implementálva (manifest-alapú)
- Augmentációs pipeline (train) és clean pipeline (val/test) definiálva
- ImageNet normalizálás alkalmazva (transfer learning előfeltétele)
- Class weights kiszámítva inverz frekvencia módszerrel (`w_c = N_train / (K × n_c)`)
- DataLoader setup: train shuffle + drop_last=True, val/test fix
- Sanity check: denormalizált batch vizualizáció

### Döntések

| Szempont | Döntés |
|----------|--------|
| Célméret | **224×224** px |
| Normalizálás | **ImageNet** mean/std – pretrained modellekhez szükséges |
| Augmentáció | RandomHorizontalFlip(p=0.5), ColorJitter(br=0.3, ct=0.3, sat=0.2, hue=0.1), RandomRotation(±15°) |
| Class weight | Inverz frekvencia: G (~1.73×), No hand (~2.59×) kapják a legmagasabb súlyt |
| Framework | PyTorch + torchvision.transforms |

### Státusz
✅ DataLoader pipeline kész, class weights számítva, sanity check vizualizáció generálva

---

## 🗓️ 2026-05-12 – Környezeti konfiguráció és hibajavítás (1. körös)

### Probléma: CUDA nem volt elérhető – vegyes conda/pip konfliktus

**Tünet:** `torch.cuda.is_available()` → `False`

**Diagnózis:** Kettős, egymásnak ellentmondó PyTorch telepítés volt a környezetben:
- `libtorch-2.10.0` conda-ból: **CPU-only** build (`cpu_openblas_h7e86a07_3`)
- `pytorch-cuda=12.4` conda-ból + `nvidia-*-cu12` csomagok pip-ből: CUDA-s runtime
- A conda a CPU-s `libtorch`-ot töltötte be, figyelmen kívül hagyva a pip-es CUDA binárisokat

**Gyökérok:** Az eredeti `environment.yaml` nem rögzítette a `pytorch-cuda` verziót, így a conda solver CPU-only `libtorch`-ot oldott fel, miközben a pip-es nvidia csomagok egymásnak ellentmondó szimbólumokat hoztak.

### Második hiba: `iJIT_NotifyEvent` undefined symbol

**Tünet:**
```
ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

**Diagnózis:** A conda-forge `intel-openmp` és a pytorch channel `mkl` verziói közötti szimbólum-konfliktus. Az `iJIT_NotifyEvent` az Intel VTune profiler része; a conda-forge verziója nem exportálta ezt a szimbólumot a PyTorch által várt formában.

**Sikertelen javítási kísérlet:**
```bash
conda install mkl=2025.0.0 intel-openmp=2025.0.0 -c defaults --override-channels -y
# Eredmény: "All requested packages already installed." – nem segített
```

### Megoldás: PyTorch teljes cseréje pip-re

```bash
# 1. Conda PyTorch eltávolítása
conda remove pytorch torchvision torchaudio pytorch-cuda pytorch-mutex torchtriton --force -y

# 2. pip-es telepítés (saját bundled MKL/OpenMP, nincs külső konfliktus)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Eredmény:**
```
PyTorch: 2.6.0+cu124
CUDA elérhető: True
GPU: NVIDIA T500
```

### `environment.yaml` frissítve
- PyTorch forrás: conda → **pip** (`https://download.pytorch.org/whl/cu124`)
- Verziók rögzítve: `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, `torchaudio==2.6.0+cu124`

### Tanulság
> Ne keverd a conda és pip PyTorch telepítést. Ha pip-pel telepíted a `torch`-ot, távolítsd el előbb a conda-verziót `--force` flag-gel, különben a conda-s `libtorch` felülírja a pip-es binárisokat.

### Státusz
✅ CUDA gyorsítás működik – `NVIDIA T500`, CUDA 12.4, PyTorch 2.6.0

---

## 🗓️ 2026-05-12 – Modell sorozat megtervezése és notebookok létrehozása

### Elvégzett műveletek

- `04_model.ipynb` átnevezve → `04c_efficientnet_b0.ipynb`
- Modell-összehasonlítási sorozat megtervezve és implementálva

### Notebook sorozat

| Notebook | Modellek | Leírás |
|----------|----------|--------|
| `04a_baseline_ml.ipynb` | SVM (HOG), SVM (CNN features), RF, KNN, LR, XGBoost | Hagyományos ML baseline HOG + ResNet50 avgpool features alapján |
| `04b_mobile_cnn.ipynb` | MobileNetV3-Small, MobileNetV3-Large, ShuffleNetV2 x1.0 | Könnyűsúlyú CNN-ek, két fázisú fine-tuning |
| `04c_efficientnet_b0.ipynb` | EfficientNet-B0 | Kétfázisú fine-tuning (átnevezett `04_model.ipynb`) |
| `04d_advanced_cnn.ipynb` | ResNet-50, EfficientNet-B3 | Nagyobb kapacitású CNN-ek, teljes összehasonlítás |
| `04e_vit.ipynb` | ViT-B/16 | Vision Transformer (opcionális, VRAM-igényes) |
| `05_model_selection.ipynb` | Ensemble + kiválasztás | Legjobb modellek összesítése |

### Ajánlott futtatási sorrend és várható accuracy

| Rang | Modell | Várható Test Acc (kis adat) |
|------|--------|---------------------------|
| 1 | EfficientNet-B3 | ~90–94% |
| 2 | ResNet-50 | ~88–92% |
| 3 | EfficientNet-B0 | ~85–92% |
| 4 | MobileNetV3-Large | ~80–88% |
| 5 | SVM + CNN features | ~75–82% |
| 6 | MobileNetV3-Small | ~78–85% |
| 7 | ShuffleNetV2 | ~75–83% |
| 8–10 | HOG-alapú ML modellek | ~60–78% |

### Státusz
✅ Mind a 4 notebook (`04a`–`04d`) elkészítve, letölthető

---

## 🗓️ 2026-05-12 – Környezeti hiba (2. körös): `CUDA unknown error`

### Probléma

**Tünet:** `torch.cuda.is_available()` → `False`, hibaüzenet:
```
UserWarning: CUDA initialization: CUDA unknown error
(Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:109.)
RuntimeError: CUDA unknown error
```

**Fontos:** `nvidia-smi` rendesen futott, a GPU látható volt OS szinten – a hiba PyTorch/kernel driver szinten volt.

### Diagnózis – okkeresés sorrendben

1. **Első gyanú:** conda-s CUDA runtime csomagok ütközése a pip-es PyTorch `nvidia-*-cu12` csomagjaival
   - `cuda-version 12.9` (nvidia csatorna) automatikusan frissíthetett, magával húzva conda-s CUDA runtime könyvtárakat
   - Megoldási kísérlet: conda CUDA csomagok eltávolítása + pip reinstall → **nem segített**

2. **Valódi gyökérok:** `nvidia_uvm` kernel modul "szennyezett" állapotba került
   - Az nvidia-smi kimenetében látható volt egy aktív Python process (PID 67453, 50MiB), amely CUDA kontextust foglalt
   - Logout **nem elegendő** – a kernel modulok betöltve maradnak, az Xorg is folyamatosan futott

### Megoldási kísérletek sorrendben

| Kísérlet | Eredmény |
|---------|---------|
| `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` | ❌ Nem működött – Xorg lefoglalta a modult |
| Logout és újra bejelentkezés | ❌ Kernel modulok megmaradtak |
| `sudo reboot` | ✅ **Megoldás** |

### Megoldás

```bash
sudo reboot
```

Reboot után:
```
PyTorch: 2.6.0+cu124
CUDA: True
GPU: NVIDIA T500
```

### Tanulság
> A `nvidia_uvm` kernel modul "frozen" állapotba kerülhet suspend/resume után, vagy ha egy Python process nem szabályosan zárta le a CUDA kontextust. Bejelentkezett KDE Plasma munkamenetben az Xorg folyamatosan foglalja a modult – ezért csak teljes reboot garantálja a tiszta CUDA inicializációt.
>
> Szerveres (headless) környezetben alternatíva: `sudo systemctl stop gdm && sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm && sudo systemctl start gdm` – de ez kilövi a grafikus munkamenetet.

### Státusz
✅ CUDA működik, fejlesztés folytatható – következő lépés: `04a_baseline_ml.ipynb` futtatása

---

*Ez a dokumentum folyamatosan bővül. Semmi nem törlődik – minden döntés és fázis visszakereshető.*
