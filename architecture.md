# Guitar Chord Recognition – Teljes Technikai Dokumentáció

**Verzió:** V14.2 Pipeline  
**Dátum:** 2026-05-21  
**Projekt:** `guitar-chord-recognition`  

---

## Tartalom

1. [Magas szintű Pipeline (V14)](#1-magas-szintű-pipeline-v14)
2. [Feature Engineering & Kinyert adatok](#2-feature-engineering--kinyert-adatok)
3. [ML Architektúra és Döntési mechanizmus](#3-ml-architektúra-és-döntési-mechanizmus)
4. [Tanítási és Export folyamat](#4-tanítási-és-export-folyamat)
5. [Deployment & Diagnosztika](#5-deployment--diagnosztika)
6. [Modulok áttekintése](#6-modulok-áttekintése)
7. [Konfiguráció referencia](#7-konfiguráció-referencia)

---

## 1. Magas szintű Pipeline (V14.1)

### V14.1 Standard

> **Bit-szintű azonosság követelmény:** minden bemeneti interfész (notebook, CLI, Streamlit) kizárólag a `src/preprocess.py::preprocess_image_input(raw_bytes, max_long_edge)` függvényen keresztül tölthet be képet. Más betöltési út (`cv2.imread`, `cv2.imdecode`, `PIL.Image.open` közvetlen) tilos, mert az EXIF-transzponálás és az interpoláció eltérő pixel-értékeket adhat → Hough/MediaPipe eredmény mismatch.

### V14.2 – Streamlit Hough-konzisztencia fix

**Probléma azonosítva:** `max_long_edge=1920` a Streamlit-ben a 4K képet ~55%-ra skálázta → Hough-vonalak száma 18→2-re esett vissza → `GEOMETRIC_RULE` fallback, nem `INTENSITY_DATA` (Sobel).

**Gyökérok:** a Hough-paraméterek (`threshold=30`, `minLineLength=15% of min(H,W)`) 4K felbontásra vannak kalibrálva. A `threshold=30` abszolút szavazatszám — kisebb képen a rövidebb vonalszakaszok kevesebb szavazatot gyűjtenek.

**Fix:** `_UPLOAD_MAX_PX = 0` → nincs resize, a Streamlit natív felbontáson fut, pontosan mint a notebook.

**Miért NEM `cv2.imdecode`?**
- `preprocess_image_input(raw_bytes: bytes, ...)` bytesot vár, nem numpy arrayt → API törne
- `cv2.imdecode` nem kezeli az EXIF-orientációt → mobil portrék elforgatva érkeznének
- A felbontási különbség volt az ok, nem a PIL-dekódolás módja

**Debug UI (V14.2):**

| Új metrika | Forrás | Cél |
|-----------|--------|-----|
| `Hough sorok (össz.)` | `len(result["lines"])` | Tipikusan 15-20 @ 4K, 2-5 @ 1920px |
| `Long / Fret sorok` | `len(split["long_lines"]) / len(split["fret_lines"])` | Nyak- vs. bund-párhuzamos vonalak |
| Span < 2000px warning | `span_px < _SPAN_WARN_PX` | Alacsony felbontás figyelmeztető |

### 1.1 Átfogó adatfolyam

```
Bemeneti kép (raw bytes / BGR)
        │
        ▼
[preprocess_image_input]   ← egyetlen engedélyezett belépési pont
  PIL megnyitás → EXIF-transpose → opcionális LANCZOS kicsinyítés
  → BGR numpy array (natív felbontás, ha max_long_edge=0)
        │
        ▼
[PNG temp fájl]  ← lossless, MediaPipe és a pipeline ezen olvas
        │
        ▼
[MediaPipe HandLandmarker]──► Landmarks (21 pont, normalizált x/y/z)
        │                      Finger Mask (kéz-pixelek elnyomása)
        │                      Anchor (csuklóvektor → nyakirány-hint)
        │
        ▼
[Step 1] Canny éldetektálás
  Gaussian Blur (5×5) → Canny(25, 80) → bináris élkép
  Finger Mask-olt verzió (edges_masked) a Hough-hoz
        │
        ▼
[Step 2] HoughLinesP (teljes képen)
  threshold=30, minLineLength=15% of min(H,W), maxLineGap=15
  Ha landmarks=None és 0 sor → _global_hough_fallback (±30°, kétmenetes)
        │
        ▼
[Step 3] Nyakirány meghatározása
  Hosszal súlyozott szöghisztogram (36 bin, −90°..+90°)
  Anchor-alapú finomítás ha < 3 hosszú vonal
  Step 3b: bund-vonalak perpendicular irányából visszaszámolt blended szög
        │
        ▼
[Step 4] Vonalak szétválasztása
  long_lines  : ±20° a nyakkal párhuzamos
  fret_lines  : ±20° a nyakra merőleges
        │
        ▼
[Step 5] Külső nyakélek detektálása (v9, outlier-alapú)
  Vetületek sorrendbe → outlier gap-teszt (2.5× outlier_ratio)
  Klaszter-centroid → 30% expansion margin ha nincs outlier
        │
        ▼
[Step 6] Trapézoid → Perspektíva Warp
  step6_clamp_trapezoid_extent  – test-oldali ROI korlátozás
  step6_trapezoid               – 4 sarokpont (TL/TR/BR/BL)
  Trapézoid sanitás: aspect ≥ 1.5, area_frac ∈ [0.004, 0.90]
  Orientáció-guard: h > w és lines_mean_angle > 35° → REJECT
  step6_warp                    – cv2.getPerspectiveTransform → (H, H_inv, canon)
  canon: 600×80 px kanonikus kép
        │
        ▼
[Step 6d] Post-warp Shear Korrekció
  HoughLinesP a kanonikus képen (közel-függőleges vonalak)
  Hosszal súlyozott α szögátlag → S shear mátrix
  Ha |α| > 0.5°: x_corr = x − tan(α)·y
  Eredmény: corrected canon + frissített H/H_inv
        │
        ▼
[Bunddetektálás – IntensityFretDetector / GeometricFretDetector]
  (részletek: 1.4. fejezet)
        │
        ▼
[Step 9] Ujjhegy vetítés
  step9_project_fingertips: landmarks → H → kanonikus x/y
  fret_est: legközelebbi predicted_x bund sorszáma
  string_norm: y / CANONICAL_H (0=fej, 1=test)
        │
        ▼
[Irány normalizálás]
  _derive_is_flipped → canon_norm = flip(canon, 1) ha szükséges
  (részletek: 1.2. fejezet)
        │
        ▼
[Low-confidence Flag]
  coverage_ratio < 0.20 VAGY warp_stretch > 2.0
        │
        ▼
Kimenet: result dict (100+ kulcs) → features.py → ML modellek
```

### 1.2 Irány-agnosztika: Nut-irány meghatározása

A rendszer **háromszintű döntési hierarchiával** határozza meg, hogy a nut a kanonikus kép bal vagy jobb oldalán van-e (`is_flipped`):

#### Elsődleges: bundtávolság-gradiens (`fit_direction`)

A `step8_fit_fret_rule` a detektált bundpozíciók szomszédos távolságainak trendjéből határozza meg az irányt:

```
trend = polyfit(arange(n_spacing), spacings, deg=1)[0]
if trend > mean(spacings) * 0.05:
    _direction = "reversed"   → nut jobb oldalon
```

A nut felé haladva a bundtávolságok növekednek (17.817-es szabály). Ha a bal→jobb irányban növekvő spacingeket látunk, az a nut jobb oldalát jelenti. Ez csak akkor megbízható, ha `coverage_ratio ≥ 0.30`.

#### Fallback: anatómiai vektor (`detect_guitar_orientation`)

```python
delta_x_norm = index_mcp.x − wrist.x
side_hint = "right" if delta_x_norm > 0 else "left"
flip_logic = side_hint == "right"
```

Ha a handedness (Left/Right kéz) ellentmond a kép-koordinátában látott oldalnak, tükörképes forrás gyanítható → `mirrored_override = True`, és az anatómiából következő oldalt veszi.

#### Utolsó mentsvár: nyers csukló vs. mutatóujj-tő x

```python
return index_mcp_x < wrist_x  # wrist further right → nut on right
```

#### `is_flipped` hatása a pipeline kimenetére

| Elem | Hatás |
|------|-------|
| `canon_norm` | `cv2.flip(canon, 1)` ha flipped |
| Group B feature | páros indexű x-komponensek negálva (`b[0::2] = -b[0::2]`) |
| Group F feature | `f[1] = -f[1]` (sin-komponens) |
| `compute_rel_fingertip_positions` | `cx_norm = CANONICAL_W − cx` ha flipped |

### 1.3 Hough-fallback: kéz nélküli mód

Ha `landmarks is None` **és** a standard `step2_hough` 0 vonalat ad, a `_global_hough_fallback` lép életbe:

**Kétmenetes stratégia:**

| Menet | threshold | minLineLength | maxLineGap | Feltétel |
|-------|-----------|---------------|------------|----------|
| Pass 1 | 20 | max(w//5, 60) | 30 | Mindig fut |
| Pass 2 | 12 | max(w//8, 40) | 50 | Ha < 2 horiz. vonal VAGY vert > horiz |

**Szűrési feltételek (mindkét menetre):**

1. **Szög-szűrés ±30°:** `abs(angle_from_horizontal) ≤ 30°` – elutasítja a bundvonalakat, csak a nyak-párhuzamos éleket tartja.
2. **Border exclusion:** elveti a vonalakat, amelyek mindkét Y-végpontja a kép felső/alsó 5%-ában van (Canny határartefaktumok).
3. **Hossz-szűrés:** csak `length ≥ 40% * longest_line` vonalak maradnak.

Ezután a trapézoid Y-kiterjedése a vonalak tényleges Y-tartományára + 15% margóra korlátozódik (`fallback_roi_clamp`), és ha az x-lefedettség < 30%, a trapézoid X-irányban kiterjed a nyakirány mentén (`fallback_horiz_extend`).

### 1.4 Bunddetektáló architektúra (Plug-and-Play)

```
FretDetectorInterface (ABC)
    ├── IntensityFretDetector   ← ALAPÉRTELMEZETT (INTENSITY_DATA)
    └── GeometricFretDetector   ← FALLBACK (GEOMETRIC_RULE)
```

#### IntensityFretDetector

**Profilstratégia auto-módban (shear-alapú döntés):**

| Feltétel | Stratégia | Indok |
|----------|-----------|-------|
| n_lines ≥ 4, residual < 0.3°, confidence ≥ 0.75 | Sobel-X | Egyenes bundok → pontos gradienscsúcsok |
| Egyéb | Max-pooling | Dőlésre-invariáns, robusztus zajos képeknél |
| Sobel SNR < 1.5 (auto-check) | Max-pooling fallback | Sobel profil gyenge → nem megbízható |

**Csúcsdetektálás:** `scipy.signal.find_peaks` dinamikus paraméterekkel:
- `dyn_prom` a shear residual és confidence függvénye
- Ha < 2 csúcs: relaxált `find_peaks` (75%-os height, 50%-os prominence, nincs width korlát)
- Ha < 2 csúcs marad: GeometricFretDetector step7 fallback

**Kézmaszk-alapú elnyomás:** a kanonikus hand_mask-ből kapott oszlopok az intenzitás-profilban 0-ra állnak → az ujjak területén nem keletkezhet bund-jelölt.

#### GeometricFretDetector

`step7_fret_lines_canonical` → HoughLinesP a 600×80-as kanonikus képen → szélességalapú klaszterszűrés → klaszterközéppontok.

#### Közös post-processing (mindkettőn)

1. `suppress_finger_pairs`: 8–22 px-en belüli páros bundjelöltek eltávolítása (ujjak okozta kétszeres detekció)
2. `step8_fit_fret_rule` – kétlépéses illesztés (pass1 → refine_frets_by_fit → pass2)

---

## 2. Feature Engineering & Kinyert adatok

### 2.1 Feature vektor szerkezete (56 dimenzió – Basic)

```
[0–41]   Group B: Wrist-normalized landmarks    (42 dim)
[42–43]  Group D: Detection flags               ( 2 dim)
[44–45]  Group F: Neck angle cos/sin            ( 2 dim)
[46–50]  Group G: Fret index per finger         ( 5 dim)
[51–55]  Group H: String norm per finger        ( 5 dim)
```

#### Group B – Legfontosabb csoport (42 dim)

**Számítás:**

```python
wrist = pts[0]
hand_scale = ||pts[9] − wrist||   # csukló–középső MCP euklideszi távolság
centered = (pts − wrist) / hand_scale
vec[:] = centered.flatten()       # 21 × [x, y] → 42 dim
```

| Tulajdonság | Érték |
|-------------|-------|
| Koordinátarendszer | Csukló-centrált, kéz-skálára normalizált |
| Dimenzió per landmark | x, y (z kihagyva) |
| Megmarad ha `ok=False` | Igen, ha landmarks ≠ None |
| Flip-korrekció | `b[0::2] = -b[0::2]` (páros indexek = x koordináták) |

**Miért domináns:** a kézalak-topológia (ujjak hajlítása, terpesztés) önmagában elegendő sok akkord megkülönböztetéséhez, a fogólappal való interakció nélkül is.

#### Group D – Detekciós flagek (2 dim)

| Index | Név | Érték |
|-------|-----|-------|
| 42 | `hand_detected` | 1.0 ha landmarks ≠ None |
| 43 | `fretboard_detected` | 1.0 ha ok=True AND coverage_ratio ≥ 0.40 |

#### Group F – Nyakszög (2 dim)

```python
vec[0] = cos(angle_rad)
vec[1] = sin(angle_rad)   # negálva ha is_flipped
```

Irány-agnosztikus kódolás: a sin előjele a nyak dőlési irányát hordozza, tükrözéskor korrektív negálással.

#### Group G – Bund-index per ujj (5 dim)

```python
g[col_i] = clip(fret_est / N_FRETS, 0.0, 1.0)
```

`fret_est` a `step9_project_fingertips`-ből jön: az ujjhegy kanonikus x-koordinátájához legközelebbi `predicted_x` kulcs (bund sorszáma 0–24).

#### Group H – Húr-pozíció (5 dim)

```python
h[col_i] = clip(string_norm, 0.0, 1.0)
```

`string_norm = canon_y / CANONICAL_H`: a húr pozíciója a ROI magassági tengelyén (0 = fej oldal, 1 = test oldal). Vízszintes tükrözéstől független.

### 2.2 Inlay-adatok (4 dim extra – Inlay verzió, összesen 60 dim)

Az inlay feature az alapvető X_basic vektort **4 extra dimenzióval** bővíti: a 3., 5., 7. és 9. bund egypontos inlay-jeinek normalizált x-pozíciója.

**Inlay elvi pozíciója** (n-1. és n. bund közötti középpont):

```python
INLAY_NORM_DICT[n] = (FRET_POS_NORM[n-1] + FRET_POS_NORM[n]) / 2.0
```

**Kinyerés a `_inlay_features()` függvényben:**

```python
for fret_n in [3, 5, 7, 9]:
    if fret_n in pred_x:
        vec[i] = clip(pred_x[fret_n] / CANONICAL_W, 0.0, 1.0)  # mért pozíció
    else:
        vec[i] = clip(FRET_POS_NORM[fret_n], 0.0, 1.0)          # elméleti pozíció
```

**Feltétel:** csak ha `fretboard_detected = 1` (coverage ≥ 0.40), különben csupa 0.

| Bund | Elvi normalizált pozíció |
|------|--------------------------|
| 3 | ≈ 0.157 |
| 5 | ≈ 0.250 |
| 7 | ≈ 0.330 |
| 9 | ≈ 0.401 |

### 2.3 Relatív ujjpozíciók (`compute_rel_fingertip_positions`)

Irány-agnosztikus koordináták az ML osztályozáshoz:

```
rel_fret_x  ∈ [0.0, 1.0]  : bunden-belüli pozíció
                              0.0 = nut-oldali bund felett
                              1.0 = test-oldali bund felett
rel_string_y ∈ [0.0, 1.0] : y-alapú húrpozíció, tükrözéstől független
```

**`rel_fret_x` számítása:**

```python
cx_norm = (CANONICAL_W − cx) if is_flipped else cx
# pred_norm is tükrözve ha flipped
for i in range(len(sorted_frets) - 1):
    if x_lo <= cx_norm <= x_hi:
        return (cx_norm - x_lo) / (x_hi - x_lo)
```

### 2.4 CNN bemenet (224×224)

```python
# canon_norm = always nut-left (flipped ha szükséges)
img_resized = cv2.resize(canon_norm, (224, 224), interpolation=cv2.INTER_AREA)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
image = img_rgb.astype(np.float32) / 255.0  # [0, 1], RGB
```

**Dinamikus szélesség (`_dyn_w`) a pixeláció elkerülésére:**

```python
_scale_h = CANONICAL_H / src_hgt      # vertikális szkalafaktor a ROI méretéből
_dyn_w = int(round(src_len * _scale_h))
_dyn_w = max(min(_dyn_w, CANONICAL_W), CANONICAL_W // 4)
_stretch = CANONICAL_W / _dyn_w

if _stretch > 1.5:
    canon_natural = cv2.resize(canon, (_dyn_w, CANONICAL_H), INTER_AREA)
    # → diagnosztikában jelenik meg, ML mindig CANONICAL_W-t kap
```

A `low_confidence` flag jelzi ha `stretch > 2.0` (erős felskálázás → pixelált bemenet a CNN-nek).

### 2.5 17.817-es bund-illesztés (`step8_fit_fret_rule`)

A bundok elméletileg a következő normalizált pozíción vannak:

```
FRET_POS_NORM[n] = 1 − (1/2)^(n/12)
```

Ez az `equal temperament` skálából következik: minden oktávban megduplázódik a frekv., minden 12 félhangban feleződik a maradék távolság a nyereg felé.

**Illesztési módszer-hierarchia:**

| Módszer | Feltétel | Leírás |
|---------|----------|--------|
| `inlay_anchored` | inlay_xs nem üres | Inlay-kandid.-alapú 2-param keresés |
| `nut_anchored` | `nut_anchored=True` és Nut detektálva | offset=0, csak scale optimalizálva |
| `ratio_run` | ≥ 3 konzisztens spacinget talál | Legkisebb 2¹/¹² arányú futam |
| `ransac_fallback` | Minden más fail | 2-pontos RANSAC scale-tartomány szűréssel |

**Irány-detekció:** a spacingek trendje (`polyfit`) dönti el, hogy a detektált x-koordinátákat eredeti sorrendben (`forward`) vagy tükrözve (`reversed`) illesztjük.

**Score:**

```python
score = coverage_ratio × nut_prior
coverage_ratio = n_matched / n_visible
nut_prior = 1.0 − 0.7 × fret_start / (N_FRETS + 1)
```

---

---

## 2b. Geometriai biztonsági korlátok (integrity v2, 2026-05-21)

### 2b.1 Bemeneti sztendardizálás (`preprocess_image_input`)

A `src/preprocess.py::preprocess_image_input(raw_bytes, max_long_edge=0)` függvény az összes bemeneti ág egységes belépési pontja.

| Lépés | Implementáció | Cél |
|-------|---------------|-----|
| PIL megnyitás | `PILImage.open(BytesIO(raw_bytes))` | Formátumtól független dekódolás |
| EXIF-transpose | `ImageOps.exif_transpose(pil_img)` | Mobilról érkező portré/landscape EXIF-forgás korrekciója; `cv2.imread` ezt nem kezeli |
| RGB konverzió | `pil_img.convert("RGB")` ha mode ≠ RGB | RGBA/P módok biztonságos kezelése |
| Arányőrző kicsinyítés | `LANCZOS` → `max(w, h) > max_long_edge` esetén | Streamlit-upload: max_long_edge=1920; notebook/pipeline: max_long_edge=0 (nincs resize) |
| BGR konverzió | `cv2.cvtColor(np.asarray(pil), COLOR_RGB2BGR)` | OpenCV-kompatibilis kimenet |

**`max_long_edge` értelmezése:**

| Hívó | max_long_edge | Hatás |
|------|---------------|-------|
| `app.py` (Streamlit) | 1920 | 4K fotókat 1920px-re csökkenti |
| `notebooks/10_interactive_orchestrator.ipynb` | 0 | Natív felbontás megmarad (4640×3488 px) |
| `src/inference.py` predict() belső temp | – | A már BGR-ré konvertált képet PNG-ként menti tempfájlba |

**Determinisztikus ekvivalencia (list vs. upload):** mindkét notebook-ág (képlista-választás és feltöltés) azonos `preprocess_image_input → cv2.imwrite(.png) → run_v14_pipeline` láncon megy át → bit-azonos kimenet garantált (igazolt: max pixel diff = 0).

**Felbontáskülönbség (V14.2 után eliminálva):** mindkét interfész `max_long_edge=0` → natív felbontás, azonos Hough-eredmény. A `🔍 Részletes Pipeline Diagnosztika` expander mindig megjeleníti a bemeneti felbontást mint sanity-check.

**Elavult:** a korábban dokumentált `_to_720p()` függvény már nem létezik; `preprocess_image_input` váltja fel teljes egészében.

### 2b.2 Trapézoid validálás – kiterjesztett szanitás

A `validate_trapezoid` (src/fretboard.py) az eredeti 3 geometriai szűrőn felül 3 új korlátot alkalmaz:

| Ellenőrzés | Küszöb (CFG kulcs) | Hatás |
|------------|-------------------|-------|
| **Minimum aspektusarány** | `sanity_min_aspect = 1.5` | Elutasítja a torony-alakú (majdnem négyzet) trapézoidokat |
| **Maximum területarány** | `sanity_area_limits.max_frac = 0.90` | Close-up képeken a 30%-os expansion margin miatt a trapéz területe > kép területe lehet; 0.90-es határ ezt engedi |
| **Minimális szélesség** | `sanity_min_width_px = 400 px` | Elutasítja a fejrész (headstock) körüli, rövid ROI-t |
| **Tower-guard** | `sanity_max_tower_ratio = 1.0` | h_trap > w_trap → torony-alak elutasítva |
| **Ujjhegy-átfedés** | `sanity_fingertip_overlap = 0.60` | Ha a kéz ujjhegyeinek < 60%-a esik a trapézon belülre → fejrész-ugrás detektálva |

**Ujjhegy-átfedés részletei:**

- Csak akkor aktív, ha `landmarks is not None` (nem-kéz módban kihagyva)
- 5 ujjhegy-landmark (MediaPipe index: 4, 8, 12, 16, 20) vizsgálata
- `cv2.pointPolygonTest` a trapézoid konturán pixel-koordinátákban
- Ha `n_inside / 5 < 0.60`: warning-ok-ra kerül (`trap_reasons`), a pipeline nem áll meg

**Megjegyzés:** A `validate_trapezoid` hibák esetén az orchestrátor (`run_v14_pipeline`) csak figyelmeztető logot ír, nem utasítja vissza a képet. A `trap_ok = False` flag diagnosztikai célú; a végső `ok` flag a pipeline sikerétől függ.

**Orientáció-guard (hard reject):**

```python
_orient_thr = float(CFG.get("sanity_trap_orient_angle_thr", 35.0))
if h > w and _mean_ang > _orient_thr:
    out["invalid_reason"] = f"trap_orientation: h>w, lines_angle={_mean_ang:.1f}°>{_orient_thr}°"
    return out   # HARD REJECT
```

| CFG kulcs | Érték | Leírás |
|-----------|-------|--------|
| `sanity_trap_orient_angle_thr` | 35.0° | Ha a trapéz magas (h>w) ÉS a Hough-vonalak átlagszöge > küszöb → portrait-irányú tévdetekció, hard reject |

Korábban ez az érték hardcoded 20°-os volt (`src/fretboard.py`-ban); a konfigurálhatóvá tett 35°-os küszöb a close-up képek enyhén dőlt eseteit is megengedi.

### 2b.3 Homográfia stabilitás és warp stretch

| Paraméter | Küszöb (CFG kulcs) | Viselkedés |
|-----------|-------------------|------------|
| `warp_stretch_factor` | `low_confidence_stretch_thr = 2.5` | Ha a canonikus képet > 2.5× kellett felskálázni → `low_confidence = True` |
| `warp_dyn_w` | `svm_min_roi_width_px = 300 px` | Ha a természetes ROI-szélesség < 300 px → `warp_roi_too_narrow = True` |

### 2b.4 SVM biztonsági fék

```python
if svm_model is not None and result.get("warp_roi_too_narrow"):
    confidence = 0.0
```

Ha `warp_dyn_w < 300 px`, az inlay- és bundle-pozíciók pixeláltak / megbízhatatlanok → az SVM konfidenciáját 0.0-ra állítja az inference.py. A CNN-t ez nem érinti (teljes képen dolgozik).

### 2b.5 Notebook vs. src/ konstansok (Phase 3 audit)

Az összehasonlítás a `10_interactive_orchestrator.ipynb` (interaktív widgetek) és a `src/` modulok között az alábbi eredménnyel zárult:

| Konstans | Notebook widget default | src/ default | Státusz |
|----------|------------------------|--------------|---------|
| `peak_height` | 0.20 (widget) | 0.12 (`IntensityFretDetector`) | ✓ src/ az autoritás; a widget exploratív |
| `step7 threshold` | — | 18 (`step7_fret_lines_canonical`) | ✓ szinkronban |
| `step7 min_len_frac` | — | 0.25 | ✓ szinkronban |
| `hough_threshold` (step2) | — | 30 (`CFG["hough_threshold"]`) | ✓ szinkronban |

Eltérés nem igényelt kódváltoztatást; a notebook widget-értékek exploratív jellegűek és nem befolyásolják a produkciós pipeline-t.

---

## 3. ML Architektúra és Döntési mechanizmus

### 3.1 Modell-típusok és kiértékelési eredmények

| Modell | Feature bemenet | Dim | Test acc | Macro F1 | Státusz |
|--------|----------------|-----|----------|----------|---------|
| **`svm_basic`** | X_basic | 56 | **88.9%** | **0.870** | **Produkciós (SVM)** |
| `svm_inlay` | X_inlay | 60 | 86.7% | 0.827 | Nem választva |
| `rf_basic` | X_basic | 56 | 84.4% | 0.811 | Kiegészítő |
| `rf_inlay` | X_inlay | 60 | 86.7% | 0.833 | Kiegészítő |
| `lr_basic` | X_basic | 56 | 77.8% | 0.797 | Baseline |
| `lr_inlay` | X_inlay | 60 | 77.8% | 0.797 | Baseline |
| **`cnn` (MobileNetV3-Large)** | X_images | 224×224×3 | **97.8%** | — | **Produkciós (CNN)** |

**Inlay feature elvetésének indoka:** az `svm_inlay` (86.7%) gyengébben teljesített, mint az `svm_basic` (88.9%). Az inlay pozíciók zajt visznek be, ha a fogólap-detekció nem tökéletes (coverage < 1.0, fallback elméleti pozíciók). A 42 dimenziós Group B önmagában robusztusabb.

**sklearn hiperparaméterek:**

| Modell | Paraméterek |
|--------|-------------|
| SVC | kernel=rbf, C=10, gamma=scale, probability=True |
| RandomForest | n_estimators=300, n_jobs=-1 |
| LogisticRegression | max_iter=2000, C=1.0 |

**Produkciós checkpoint-ek:**

| Fájl | Modell | Test acc |
|------|--------|----------|
| `checkpoints/best_mobilenet_v3_large_phB.pth` | MobileNetV3-Large (Phase B fine-tune) | 97.8% |
| `checkpoints/best_ml_model.pkl` | SVM_basic (Group B, 42 dim) | 88.9% (→ 91.1% augmentált adaton) |

### 3.2 Osztályok

```python
CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "No hand"]
```

8 osztály: 7 gitár-akkord + 1 „nincs kéz" állapot.

### 3.3 Döntési logika: mit „lát" a modell?

#### sklearn modellek (Group B domináns)

A 42 dimenziós wrist-normalized koordináta a kézalak **topológiáját** kódolja: ujjak hajlítása, relatív elhelyezkedése. Az SVM rbf kernele a kézalak-térbeli klasztereket tanulja meg. A Group G/H (fogólap-pozíció) csak akkor ad információt, ha a fretboard detektálva van.

**Mit nem lát:** a pontos húr-pozíciót, ha `fretboard_detected = 0`. Ezért a Group B önmagában is erős, mert az anatómiai kézalak-különbség (pl. G vs. C akkord) megjelenik az ujjhegyek relatív szögében.

#### CNN (MobileNetV3)

A 224×224-es `canon_norm` képen **globális vizuális mintázatokat** tanul: bundvarratok mintázata, húrok, ujjak árnyéka, fogólapon látható reflexiók eloszlása. Az ImageNet-pretraining low-level textúra-detektorokat ad, amelyeket a fine-tuning domain-specifikus mintákra hangol.

**Mit nem lát:** explicit landmark-koordinátákat; a kéz topológiáját csak a képen látható vizuális nyomokból következteti ki.

### 3.4 Konfidencia és `low_confidence` flag

```python
low_confidence = (
    coverage_ratio < 0.20    # kevés bund illeszkedett
    OR warp_stretch > 2.0    # extrém felskálázás → pixelált CNN-bemenet
)
```

| Feltétel | Értelmezés |
|----------|------------|
| `coverage_ratio < 0.20` | A látható bundoknak < 20%-a illeszkedett a 17.817 szabályra → bizonytalan fogólapdetekció |
| `warp_stretch > 2.0` | A forrás ROI < 50% a kanonikus szélességnek → CNN pixelált, upscaled képet kap |

Az `inference.py` a pipeline failure esetén (`ok=False`) a confidence-t explicit 0.0-ra állítja, függetlenül a modell softmax kimenetétől.

---

## 4. Tanítási és Export folyamat

### 4.1 Adatgenerálás (`dataset_generator.py`)

**Bemenetek:**
- `data/split_manifest.csv` – minden képhez: `path`, `class`, `split` (train/val/test)

**Futtatás:**
```bash
python -m src.dataset_generator [--output-dir data/features]
```

**Kimenet (`data/features/`):**

| Fájl | Alak | Dtype | Tartalom |
|------|------|-------|----------|
| `X_basic.npy` | (N, 56) | float32 | `assemble_feature_vector` kimenete |
| `X_inlay.npy` | (N, 60) | float32 | X_basic + 4 inlay dim |
| `X_images.npy` | (N, 224, 224, 3) | float32 | Normalizált RGB kanonikus képek [0,1] |
| `y.npy` | (N,) | int64 | Osztálycímke-index |
| `splits.npy` | (N,) | str | 'train'/'val'/'test' |
| `class_names.npy` | (K,) | str | Index → osztálynév leképezés |

**Belső pipeline-elnyomás:** a batch futtatás alatt a `_SUPPRESS_TOKENS` lista alapján elnémítja a részletes debug printeket (`outer_edges`, `trapezoid_v9`, stb.), hogy az előrehaladás olvasható maradjon.

**Memóriakezelés:** minden kép feldolgozása után `del result, payload` → a 224×224×3 float32 képek nem halmozódnak a memóriában a hurok alatt.

### 4.2 Tanítás (`train_models.py`)

```bash
python train_models.py [--no-cnn] [--epochs-a 15] [--epochs-b 15]
```

#### sklearn modellek tanítása

```
StandardScaler → fit(X_tr) → transform(X_tr, X_va, X_te)
Classifier.fit(X_tr_scaled, y_tr)
→ Validációs metrikák: acc, macro F1
→ Pickle mentés: {'model': Pipeline, 'classes': [...], 'feature_set': 'basic'/'inlay'}
```

**Verziókezelés:** `_next_version()` megkeresi a legmagasabb `*_v<N>.pkl` számot, és `v<N+1>`-et ment.

**StandardScaler szerepe:** a különböző feature csoportok eltérő skálákon vannak (Group B: ≈ −1..+1 normalizált koordináták; Group G: 0..1; Group F: −1..+1 szög). A StandardScaler minden dimenzióban $\mu=0$, $\sigma=1$-re normalizál, így az SVM rbf kernel és az LR optimalizálás nem szenved skála-torzítástól.

#### CNN tanítás – Transfer Learning (Phase A → Phase B)

```
ImageNet pretrained MobileNetV3-Large
        │
        ▼ Phase A (frozen backbone, ~15 epoch)
   Csak model.classifier.parameters() tanítható
   optimizer: AdamW(lr=1e-3)
   EarlyStopping(patience=7) → legjobb checkpoint mentése
        │
        ▼ Phase B (fine-tune, ~15 epoch)
   model.features FELOLVAD
   AdamW differential LR:
     features: lr=1e-5   ← backbone (kis LR, ne felejtse el az ImageNet tudást)
     classifier: lr=1e-4  ← head (nagyobb LR)
   EarlyStopping(patience=7) → checkpoint felülírása
        │
        ▼
   model.state_dict() → models/cnn_v<N>.pth
```

**Miért Phase A → B?** Előbb csak a head alkalmazkodik az új (N=8) osztályozáshoz, majd az egész hálózat finomhangolódik. Ez megakadályozza, hogy a véletlenszerű head-gradiensek tönkretegyék az ImageNet-rezidenseket az első lépésekben.

**Adatformátum:** `(N, H, W, 3) → transpose(0, 3, 1, 2) → (N, 3, H, W)` TensorDataset.

---

## 5. Deployment & Diagnosztika

### 5.1 Streamlit app (`app.py`)

```bash
streamlit run app.py
```

**Háromrétegű nézet:**

| Réteg | Mindig látható? | Tartalom |
|-------|-----------------|----------|
| **Eredmény metrikák** | Igen | Akkord, Confidence, Pipeline OK/FAIL |
| **Pipeline Debug Expander** | Igen | Nut X, Shear°, Span px, Stretch, Coverage, Is Flipped, bemeneti felbontás, teljes result dict JSON |
| **Consumer** (Diagnostic OFF) | Csak ha OFF | canon_norm kép + top-3 sávdiagram |
| **Advanced Diagnostic** (Diagnostic ON) | Csak ha ON | 16-paneles pipeline audit (matplotlib figura) |

**Modell-választás (sidebar):**

| Opció | Modell | Pontosság |
|-------|--------|-----------|
| CNN – MobileNetV3-Large | `best_mobilenet_v3_large_phB.pth` | 97.8% test acc |
| SVM – 42-dim features | `best_ml_model.pkl` | 91.1% test acc |

**Caching stratégia:**
- `@st.cache_resource` – modellek egyszer töltődnek be
- `@st.cache_data(image_bytes, use_cnn)` – azonos kép + modell kombináció nem futtatja újra a pipeline-t

**Inferencia flow (`src/inference.py`):**

```
image_bytes
    │
    ▼
preprocess_image_input(image_bytes, max_long_edge=1920)  ← app.py hívja
    │  PIL + EXIF-transpose + opcionális LANCZOS kicsinyítés → BGR ndarray
    ▼
cv2.imwrite(tempfile.png)   ← lossless, MediaPipe ezen olvas
    │
    ▼
run_v14_pipeline({"path": tmp_path, "class": "?"})
    │
    ▼
PipelineVisualizer.draw_fretboard_overlay + draw_landmarks
    │
    ▼
_classify_cnn(image_bgr, model)   OR   _classify_svm(pipeline_result, model)
    │
    ▼
InferenceResult(chord, confidence, top3, ok, coverage, pipeline_result, overlay_bgr)
```

**Fontos:** a CNN az eredeti `image_bgr`-t kapja (nem a canon_norm-ot), a saját `get_transforms("val")` ImageNet normalizációjával. Az SVM a pipeline result `assemble_feature_vector`-ából csak az első 42 dimenziót (Group B) veszi (`GROUP_B_SIZE = 42`).

**PNG temp fájl indoka:** a `cv2.imwrite(.jpg)` JPEG-veszteséges tömörítése pixel-értékeket tol el, ami MediaPipe landmark-koordináta eltéréseket okoz. PNG (lossless) eliminálja a mismatch-et.

**Pipeline Debug Expander (`🔍 Részletes Pipeline Diagnosztika`):**

A Streamlit UI-ban mindig elérhető (összecsukott, de nem kell diagnosztikai módot aktiválni):

| Debug érték | Forrás a result dict-ben | Cél |
|-------------|--------------------------|-----|
| Nut X (px) | `result["nut"]["nut_x"]` | Nut detekció konzisztencia |
| Shear angle (°) | `result["shear"]["shear_angle_deg"]` | step6d shear-korrekció; ingadozás (pl. −0.59°↔−2.32°) jelzi a felbontáskülönbséget |
| Span (px) | `norm(trap["corners_px"][1] − trap["corners_px"][0])` | TL→TR él hossza; felbontás-arányos |
| Warp stretch | `result["warp_stretch_factor"]` | >2.0 → low_confidence |
| Bemeneti felbontás | `preprocess_image_input` kimenet `.shape` | Sub-pixel alignment audit: Streamlit 1920px, notebook natív |
| Teljes result dict | `_to_json(result.pipeline_result)` | Nagy tömbök (H, canon, stb.) shape-stringgé alakítva |

### 5.2 Vizualizációs réteg (`viz_diagnostics.py`) – 16 panel

```
┌──────────────┬────────────────┬────────────────┬──────────────────┐
│ Sor 1        │ Előkészítés    │                │                  │
│ P1: Eredeti  │ P2: Finger     │ P3: Trapézoid  │ P4: Warped ROI   │
│ + Landmarks  │ Mask           │ overlay        │ (kézzel)         │
├──────────────┼────────────────┼────────────────┼──────────────────┤
│ Sor 2        │ Geometria (F1) │                │                  │
│ P5: Hand     │ P6: Pre-shear  │ P7: Post-shear │ P8: Hough vonalak│
│ Mask (canon) │ ROI            │ ROI            │ + nyakszög       │
├──────────────┼────────────────┼────────────────┼──────────────────┤
│ Sor 3        │ Detekció (F2)  │                │                  │
│ P9: Sobel-X  │ P10: Masked    │ P11: Proto Nut │ P12: Debug info  │
│ canonical    │ Profile+Peaks  │ detekció       │ szövegpanel      │
├──────────────┼────────────────┼────────────────┼──────────────────┤
│ Sor 4        │ Eredmény       │                │                  │
│ P13: Final   │ P14: Canonical │ P15: Fingertip │ P16: Összefoglaló│
│ overlay      │ + fret grid    │ kanonikus       │ metrikák         │
└──────────────┴────────────────┴────────────────┴──────────────────┘
```

**Panel leírások:**

| Panel | Tartalom |
|-------|----------|
| P1 | Eredeti kép BGR→RGB, MediaPipe landmark pontok + csuklóvektor, detektált Hough-vonalak |
| P2 | Ujj-maszk (piros tint) az eredeti képen |
| P3 | Trapézoid sarokpontok az eredeti képen |
| P4 | Warped ROI (600×80), ha stretch > 1.5 → canon_natural |
| P5 | Kanonikus kézmaszk (hand_mask, piros tint) |
| P6 | canon_pre_shear – shear-korrekció előtti állapot |
| P7 | canon – shear-korrigált kanonikus kép |
| P8 | Hough-vonalak szög-gradiens színkódolással (RdYlBu) |
| P9 | Sobel-X intenzitás kanonikus kép oszloponként |
| P10 | Masked intensity profil + detektált csúcsok + predicted_x pozíciók |
| P11 | Prototype nut detektálás (ha elérhető) |
| P12 | Debug szövegpanel: ok, invalid_reason, coverage, stretch, flip, fret_detector_method |
| P13 | Eredeti kép + fogólap-overlay (trapézoid + bund-rácsok H_inv-vel visszavetítve) |
| P14 | Kanonikus kép + bund-rácsok + ujjhegyek |
| P15 | Ujjhegyek kanonikus koordinátái + rel. pozíciók |
| P16 | Összefoglaló szöveges riport: osztály, confidence, coverage, is_flipped, n_frets |

**Automatikus mentés:** minden audit-kép PNG-ként kerül `output/10_interactive_orchestrator/` mappába, az `{fname}_{timestamp}_diag.png` névkonvencióval.

---

## 6. Modulok áttekintése

| Modul | Felelősség |
|-------|------------|
| `src/config.py` | CFG dict (pipeline paraméterek), PATHS, NUT_CONSTRAINTS |
| `src/constants.py` | CANONICAL_W/H, FRET_POS_NORM, INLAY_NORM_DICT, FINGER_TIP_IDX |
| `src/geometry.py` | Step 1–8 implementációk (Canny → Hough → trapézoid → warp → bundle-illesztés) |
| `src/fretboard.py` | `run_v14_pipeline` orchestrátor, FretDetectorInterface + implementációk, `validate_trapezoid`, `_derive_is_flipped`, `_global_hough_fallback` |
| `src/hand_landmark.py` | MediaPipe HandLandmarker singleton, `step9_detect_landmarks`, `step9_project_fingertips`, `build_finger_mask`, anchor-alapú nyakszög |
| `src/features.py` | `assemble_feature_vector` (56 dim), `compute_rel_fingertip_positions`, `get_ml_ready_payload`, batch extrakció |
| `src/dataset_generator.py` | `export_dataset` (X_basic/inlay/images + y/splits/class_names npy) |
| `src/dataset.py` | PyTorch Dataset + `get_transforms` (ImageNet normalizáció) |
| `src/models.py` | `build_model` (MobileNetV3-Small/Large, EfficientNet-B0/B3, ResNet50, ShuffleNet), `freeze_backbone`, `unfreeze_last_blocks` |
| `src/train.py` | `EarlyStopping` callback |
| `src/inference.py` | `predict` orchestrátor, `load_cnn/svm`, `InferenceResult` dataclass |
| `src/viz.py` | `PipelineVisualizer` (overlay rajzolás) |
| `src/viz_diagnostics.py` | `create_full_pipeline_audit` (16-panel matplotlib figura) |
| `src/prototype_nut_detector.py` | Kísérleti nut/inlay detektálás (diagnosztikában) |
| `src/preprocess.py` | `preprocess_image_input(raw_bytes, max_long_edge)` — egységes bemeneti előkészítő (EXIF-transpose, LANCZOS kicsinyítés, BGR konverzió); `ImagePreprocessor` — CLAHE + opcionális blur preprocessor lánc |
| `src/roi.py` | ROI segédfüggvények |
| `src/logic.py` | Magasabb szintű orchestrációs logika |
| `train_models.py` | CLI belépési pont sklearn + CNN tanításhoz |
| `app.py` | Streamlit frontend |

---

## 7. Konfiguráció referencia

### Kanonikus tér

| Paraméter | Érték | Leírás |
|-----------|-------|--------|
| `canonical_w` | 600 px | Kanonikus kép szélessége |
| `canonical_h` | 80 px | Kanonikus kép magassága |
| `n_frets` | 24 | Modellezett bundok száma |
| `fret_rule` | 17.817 | Equal temperament szabály (oktáv-osztó) |

### Bunddetekció

| Paraméter | Érték | Leírás |
|-----------|-------|--------|
| `fret_engine` | INTENSITY_DATA | Alapértelmezett motor |
| `step8_tol_px` | 12.0 | Illesztési tolerancia (px) |
| `step8_ratio_tol` | 0.10 | Spacing-ratio tolerancia a 2¹/¹²-höz |
| `fret_refine_enabled` | True | Kétlépéses post-fit finomítás |
| `fret_refine_tol_px` | 12.0 | Refine tolerancia (px) |

### Geometriai szanitás

| Paraméter | Érték | Leírás |
|-----------|-------|--------|
| `sanity_min_aspect` | 1.5 | Minimum aspektusarány (w/h) |
| `sanity_area_limits.min_frac` | 0.004 | Minimum területarány (trapézoid / képterület) |
| `sanity_area_limits.max_frac` | 0.90 | Maximum területarány |
| `sanity_max_edge_angle_diff_deg` | 20.0° | Trapézoid oldal-szög eltérés max. |
| `sanity_trap_orient_angle_thr` | 35.0° | Orientáció-guard: h>w és lines_angle > küszöb → hard reject |
| `sanity_min_width_px` | 400 px | Minimum trapézoid-szélesség (px) |
| `sanity_max_tower_ratio` | 1.0 | Tower-guard: h/w > 1.0 → elutasítva |
| `sanity_fingertip_overlap` | 0.60 | Min. ujjhegy-átfedés a trapézon belül |

### Konfidencia határok

| Paraméter | Érték | Leírás |
|-----------|-------|--------|
| `low_confidence_cov_thr` | 0.20 | Coverage threshold |
| `low_confidence_stretch_thr` | 2.0 | Warp stretch threshold |
| `COVERAGE_THRESHOLD` (features.py) | 0.40 | Fretboard detected flag |

### CNN tanítás

| Paraméter | Érték |
|-----------|-------|
| `img_size` | 224 px |
| `batch_size` | 16 |
| `lr_phase_a` | 1e-3 |
| `lr_phase_b_head` | 1e-4 |
| `lr_phase_b_backbone` | 1e-5 |
| `epochs_a` | 20 |
| `epochs_b` | 25 |
| `patience` | 7 |
| `num_classes` | 8 |
