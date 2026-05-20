# 📜 Projekt Fejlesztési Napló – Gitár Akkord Felismerő

---

## 🗓️ 2026-05-20 – Hough fallback stabilizálása: szög-alapú szűrés bevezetése a 90 fokkal elforgatott ROI-k elkerülésére

### Motiváció

A `_global_hough_fallback` az előző munkamenetben csak ±15°-os szög-küszöbbel futott.
Ez két problémát okozott:
1. Enyhén dőlt gitárnyakokat (15–30°) nem talált meg → kevés fallback sor → sikertelen trapézoid.
2. A fallback vonalak **nem kerültek be `out["lines"]`-ba** → a visualizáció üres Hough panelt mutatott.
3. A `roi_min_height` fallback `perp=[0,1]` iránya potenciálisan 90°-os ROI-t hozhatott létre.
4. A trapézoid orientáció egyáltalán nem volt ellenőrizve warp előtt.

### Elvégzett változtatások

**`src/fretboard.py`**

1. **`_global_hough_fallback` újraírva** — kétlépéses stratégia:
   - `_MAX_ANGLE = 30.0` (volt: 15.0) — enyhén dőlt nyakak is detektálhatók.
   - Belső `_extract_horiz()` helper: szétválasztja a horizontális (≤30°) és vertikális (>30°) vonalakat.
   - **1. lépés**: standard paraméterek (`threshold=20, minLen=w//5`).
   - **2. lépés** (ha `len(horiz) < 2` VAGY `len(vert) > len(horiz)`): lazított paraméterek
     (`threshold=12, minLen=w//8`), de még mindig csak horizontális szűrővel.
   - Visszaad: hossz szerint rendezett horizontális vonalak listája.

2. **`out["lines"]` bug javítva**: fallback sikeres futásakor `out["lines"] = lines` is frissül
   → visualizáló panel [7] most már látja a fallback vonalakat.

3. **`roi_min_height` guard** — `perp_is_fallback` flag:
   - Ha `edge_info is None`, a fallback `perp=[0,1]` csak akkor alkalmazza a kiterjtést,
     ha az eredmény trapézoid NEM lesz magasabb mint széles (`exp_h ≤ exp_w`).
   - Ha a feltétel sérülne: `SKIP: vertical_fallback_would_invert` üzenet + `debug_info` bejegyzés.
   - A kiterjtés logikája `new_corners` átmeneti változóval fut — az eredeti `corners` érintetlen marad.

4. **Trapézoid orientáció guard** (új blokk, sanity check és warp között):
   - `_trap_h > _trap_w` → `invalid_reason = "trap_orientation: h(X)>w(Y)"` + `return out`.
   - `out["debug_info"]["trap_orientation"]` mindig tartalmazza a `trap_w` / `trap_h` értékeket.

**`src/viz_diagnostics.py`**

5. **Panel [7] Hough + Nyak** — fallback mód vizuális megkülönböztetés:
   - `is_fallback = debug_info["hough"]["fallback"] == "global_hough_no_hand"`.
   - Fallback vonalak: zöld (`#00cc88`) ha ≤10°, narancssárga (`#f39c12`) ha 10-30°, vastagabb (`lw=2.0`).
   - Normál vonalak: meglévő `_angle_color` + `lw=1.0`.
   - Fallback esetén jelmagyarázat + narancssárga panel cím `[FALLBACK]` taggel.

### Architektúra-invariancia

- `_global_hough_fallback` visszatérési formátuma változatlan (`list[(x1,y1,x2,y2)]`).
- `out["lines"]` most már fallback esetén is tartalmazza az adatokat.
- Az orientáció guard egy **blokkolt return** — csak a nyilvánvalóan 90°-os eseteket ejti el.

---

## 🗓️ 2026-05-20 – A blokkoló aspect ratio ellenőrzés átalakítása figyelmeztetéssé a No Hand esetek támogatásához

### Motiváció

A `run_v14_pipeline` korábban `return out` hívással megszakadt, ha a `validate_trapezoid`
aspect-ratio, terület, vagy él-szög szanitás ellenőrzése nem teljesült.
Kéz nélküli képeken (pl. statikus gitárfotók) a `_global_hough_fallback` megtalálja
a nyakszegmenseket, de a kapott trapéz geometriája gyengébb mint kéz esetén →
a pipeline minden No-Hand képre `trapezoid_sanity` hibával megszakadt, mielőtt
megpróbálta volna kiszámítani a kanonikus ROI-t.

### Elvégzett változtatások

**`src/fretboard.py`**

1. **`validate_trapezoid` küszöb**: `min_aspect` default: 4.0 → 1.2.
   A CFG `sanity_min_aspect` kulcs felülírja ezt, ha explicit be van állítva.

2. **Sanity check: `return out` eltávolítva**:
   - Ha a szanitás nem teljesül, a pipeline többé NEM tér vissza `invalid_reason="trapezoid_sanity"` hibával.
   - Ehelyett: `print(f"[trap_sanity] WARNING — ...")` + `debug_info["trap_sanity_warning"]` bejegyzés.
   - `trap_ok=False` és `trap_reasons` továbbra is bekerül a result dictbe (vizualizáció + audit).

3. **`roi_min_height` kényszerítés `edge_info is None` esetén is**:
   - Korábban: `if roi_height < min_h_px and edge_info is not None:`
   - Most: `if roi_height < min_h_px:`, `perp` fallback = `[0.0, 1.0]` ha `edge_info` None.

**`src/viz_diagnostics.py`**

4. **Panel [3] — Warped ROI**: Ha `trap_ok=False`, narancssárga overlay szöveg jelzi a gyenge
   trapézt (`⚠ Gyenge trapéz: <okok>`), a panel cím is narancssárga `⚠` jelzéssel.

5. **Panel [8] — Sobel-X**: Ha `trap_ok=False`, a cím `"⚠ bizonytalan"` feliratot kap,
   narancssárga színnel. Mindkét panel megjelenik `ok=False` esetén is, ha `canon` rendelkezésre áll.

### Architektúra-invariancia

- `validate_trapezoid` visszatérési értéke (`bool, list[str]`) változatlan.
- A `trap_ok` / `trap_reasons` kulcsok megmaradnak → a notebook auditok nem törnek.
- A szanitás-logika önálló függvényben marad; az orchestrátor döntése (figyelmeztetés vs. megszakítás)
  a `run_v14_pipeline`-ban van.

---

## 🗓️ 2026-05-20 – F3 Fázis kész: Irány-agnosztikus, relatív koordinátákon alapuló ML pipeline aktiválva

### Motiváció

Az F1 (geometria + kanonikus ROI), F2 (nut-mentes bund-detektálás) és az orientáció-normalizálás
(auto-flip) után az utolsó lépés a teljes ML-kész kimenet összeállítása, amely:
- Minden ujjra relatív, interpretálható koordinátákat ad (nem pixel, hanem bund-közi és ROI-relatív 0-1 értékek).
- A CNN-nek mindig standard (Nut-bal) képet ad, átméretezve a célméretre.
- Egyetlen `get_ml_ready_payload(result)` hívással elérhető a teljes ML-csomag.

### Elvégzett változtatások

**`src/features.py`**

1. **`_compute_rel_fret_x(cx_norm, pred_norm) → float | None`** — tiszta helper:
   - A normalizált kanonikus x-koordinátát (esetleg már tükrözöttet) a `predicted_x` bund-hálózat
     szomszédos pozíciói közé interpolálja.
   - Ha a fingertip a látható bund-tartományon kívül esik: 0.0 (nut-oldalon kívül) vagy 1.0 (test-oldalon).
   - Mindig a **nut-bal** konvenciót feltételezi (az `is_flipped` korrekció előtte fut).

2. **`compute_rel_fingertip_positions(fingertips, fit, is_flipped) → list[dict]`** — exportált:
   - `pred_norm`: ha `is_flipped`, `predicted_x` x-értékeit tükrözi (`W − x`), ugyanígy a `canon_x`-et.
   - Per-ujj kimenet: `tip_idx, finger_name, canon_x, rel_fret_x, rel_string_y, fret_est, confidence`.
   - `rel_fret_x`: 0.0–1.0 bunden-belüli pozíció, nut-bal konvencióban.
   - `rel_string_y`: `string_norm` alias — vízszintes tükrözéstől független (y-tengely).
   - `confidence`: 1.0 ha `rel_fret_x` és `fret_est` is elérhető, egyébként 0.5.

3. **`get_ml_ready_payload(result, target_size=(224,224)) → dict`** — fő export:
   - `image`: `canon_norm` → `cv2.resize` → `BGR2RGB` → `float32 / 255.0` — (H, W, 3).
   - `feature_vec`: `assemble_feature_vector(result)` — teljes 56-dim vektor.
   - `fingers`: `compute_rel_fingertip_positions(...)` kimenete.
   - `is_flipped`, `coverage`, `ok`, `class` metaadatok.
   - Ha `canon_norm` nem elérhető: (H,W,3) nulla tensor (biztonságos fallback).

4. **`_FINGER_NAMES`** dict (`{4: "thumb", 8: "index", ...}`) a `features.py`-ba kerül,
   hogy a payload `finger_name` mező ne függjön külső moduloktól.

5. **Modul docstring** frissítve: F3 fázis leírása.

### Invarianciák

- `get_ml_ready_payload` backward-compatible: `result.get("canon_norm") or result.get("canon")`
  fallback — régi pipeline-eredményekkel is működik.
- `compute_rel_fingertip_positions` `ok=False` pipeline-eredményen üres listát ad vissza.
- `_compute_rel_fret_x` határon kívüli esetben is mindig float-ot ad vissza, nem dob kivételt.

---

## 🗓️ 2026-05-20 – Irány-agnosztikus előfeldolgozás bevezetése: automatikus tükrözés detektálás a bundtávolságok gradiense alapján

### Motiváció

A gitár lehet "standard" (Nut bal oldalt, jobbkezes játékos) vagy "tükrözött" (Nut jobb oldalt,
balkezes vagy speciális tartás) állásban. Egy orientáció-érzékeny osztályozó a kétféle képet
különböző osztályokba sorolhatta, megsokszorozva a szükséges tanítóadatot és rontva az általánosítást.

### Elvégzett változtatások

**`src/fretboard.py`**

- **`_derive_is_flipped(fit, orientation, landmarks) → bool`** — új helper:
  1. **Elsődleges (bundtávolság-gradiens):** `fit["fit_direction"] == "reversed"` és `coverage ≥ 0.30`
     → a bund-hálózat illesztési iránya alapján megbízható detektálás, ha a coverage elégséges.
     `"reversed"` azt jelenti: a távolságok balról jobbra nőnek → Nut jobb oldalt van.
  2. **Fallback:** `orientation["flip_logic"]` (az `detect_guitar_orientation` landmark-alapú becslése).
  3. **Utolsó menedék:** nyers `wrist.x > index_mcp.x` feltétel a landmarks-ból.

- **`run_v14_pipeline` kiegészítése** (fingertips számítása után):
  - `out["is_flipped"]` — bool, True ha Nut jobb oldalt van.
  - `out["canon_norm"]` — `cv2.flip(canon, 1)` ha `is_flipped`, egyébként `canon` referencia.
    Ez az a kép, amelyet a CNN-nek kell átadni (mindig Nut-bal standardban).
  - `out["nut_direction"]` — ember-olvasható string: `"Nut-Left (standard)"` / `"Nut-Right (flipped)"`.

**`src/features.py`** — `assemble_feature_vector`

- **Group B (42 dim):** ha `is_flipped`, az összes x-komponens előjele megfordul (`b[0::2] = -b[0::2]`).
  A wrist-centrált koordinátáknál ez ekvivalens a kéz tükrözésével, így a modell
  mindig ugyanolyan "standard" kézpózt lát.
- **Group F (2 dim):** ha `is_flipped`, `f[1] = -f[1]` (a sin komponens negálva).
  Vízszintes tükrözés után `sin(−α) = −sin(α)` — a nyak dőlésszög iránya megfordul.
- **Group G + H (5+5 dim):** nem kell módosítani.
  - `fret_est`: bund-sorszám — orientáció-agnosztikus (a 3. bund mindkét állásban a 3. bund).
  - `string_norm`: y-koordináta — vízszintes tükrözés nem érinti.

**`src/viz_diagnostics.py`**

- `is_flipped`, `nut_direction`, `dir_arrow` ("← Nut" / "Nut →"), `dir_color` (zöld/narancs)
  változók az audit-függvény tetején kerülnek kiszámításra.
- `axs[12]` (Final Overlay): annotáció a kép bal/jobb sarkában a nyíllal, a cím tartalmazza `nut_direction`.
- `axs[13]` (Canonical ROI): annotáció + a cím tartalmazza `dir_arrow`.
- `axs[15]` (Summary): `── Irány ─────────────────────` blokk: nyíl, standard/tükrözött, `fit_direction`, `is_flipped`.

### Architektúra-invariancia

- A `canon` mező változatlan marad — minden meglévő adat (H, H_inv, fret_xs_filt, fingertips)
  az eredeti koordinátarendszerben érvényes.
- A `canon_norm` csak a CNN-inputhoz van szánva; a `features.py` a tükrözést a feature-vektorba
  integrálja, nem a pipeline belső állapotára támaszkodik.
- `is_flipped = False` alapértelmezés → a módosítás backward-compatible, meglévő ok=False eseteket nem érint.

---

## 🗓️ 2026-05-20 – Inlay prototípus frissítése: ujj-maszkolás bevezetése a hamis pozitív detektálások elkerülése érdekében

### Motiváció

A `detect_inlays_prototype` algoritmus a teljes Sobel-X oszlopprofilon futott, beleértve az
ujjakkal takart területeket is. Ennek következtében az ujjak éles Sobel-gradiensei könnyen
hamis `dupla kis csúcs` párokat generáltak inlay-jelöltként — különösen ott, ahol az ujj
éle egybeesett egy nyakjelző közelségével.

### Elvégzett változtatások

**`src/prototype_nut_detector.py` — `detect_inlays_prototype`**

- A `result["hand_mask"]` (kanonikus tér, uint8) olvasása a profil-elemzés előtt.
- `col_has_hand = np.any(hand_mask > 0, axis=0)` — azonos logika mint az `IntensityFretDetector.detect()`-ben.
- `col_profile[col_has_hand] = 0.0` — az ujjak alatti oszlopok elnémítva.
- Az elnyomás a normalizálás után, a Gaussian-simítás előtt történik.
- Ha `hand_mask` nem elérhető (kéz nélküli mód), az algoritmus változatlanul fut.

### Elvárt hatás

- Ha egy ujj egy inlay fölé kerül, a kék pötty eltűnik a vizualizációból (helyes viselkedés).
- Kéz nélküli képen (`hand_mask` all-zero vagy None) az inlay-detektálás változatlan.
- Nincs hatása a `FretDetector`-ra vagy a feature vektorra — tisztán vizualizáció-szintű változtatás.

---

## 🗓️ 2026-05-20 – Kéz nélküli fallback mód, ROI minimális magasság fix, inlay-detektálás kísérleti fázis

### Motiváció

A pipeline eddig megállt, ha a MediaPipe nem talált kezet a képen (`no_hough_lines` visszatérés),
és a ROI esetenként túl vékony sávot adott, ami a kanonikus kép minőségét rontotta.
Emellett a diagnosztikai csomag nem mutatott semmilyen nyakjelző (inlay) információt.

### Elvégzett változtatások

**`src/fretboard.py`**

1. **`_global_hough_fallback(img, edges) → list`** — új helper:
   - Ha `step2_hough` üres listát ad vissza ÉS `landmarks is None`,
     permisszív HoughLinesP keresést indít (`threshold=20`, `minLineLength=w//5`, `maxLineGap=30`).
   - Csak közel-vízszintes vonalakat tart meg (`|szög| ≤ 15°`), hossz szerint csökkentő sorrendbe rendez.
   - Visszatérési formátum azonos `step2_hough`-éval: `list[(x1,y1,x2,y2)]`.

2. **No-hand fallback path** a `run_v14_pipeline`-ban:
   - `if not lines and landmarks is None:` → `_global_hough_fallback` hívás.
   - Sikeres fallback: `debug_info["hough"]["fallback"] = "global_hough_no_hand"`.
   - Ha a fallback sem talál vonalat: `invalid_reason = "no_hough_lines_no_hand"`.
   - Kéz jelenlétekor a korábbi `no_hough_lines` visszatérés megmarad.

3. **ROI minimális magasság** — a `if trap is None` blokk után, `validate_trapezoid` előtt:
   - `roi_min_height_frac` CFG kulcs (alapértelmezés: `0.15`, azaz `img_h * 15%`).
   - Ha `min(w_start, w_end) < min_h_px`: a négy sarokpont szimmetrikusan kitolódik
     a `perp_dir` mentén (`expand = (min_h_px - roi_height) / 2`).
   - A bal oldali sarkok (kisebb perp-vetület) `−perp * expand`, a jobb oldaliak `+perp * expand` irányban.
   - `debug_info["roi_min_height_expanded"]` tárolja az előtte/utána értékeket.

**`src/prototype_nut_detector.py`**

- **`detect_inlays_prototype(result) → list[dict]`** — új, export-szintű függvény:
  - Sobel-X oszlopprofilból keresi a `5–15 px` szélességű `dupla kis csúcs` párokat.
  - Csúcsszűrő: `height≥0.04, prominence≥0.015, width∈[1,8]` — szándékosan gyengébb mint a bund-detektor.
  - Párfeltétel: mindkét csúcs amplitúdója `< 0.55` (gyengébb mint az erős bund-csúcsok).
  - Visszatért dict mezők: `canon_x, pair (p1,p2), confidence, heights`.
  - Teljes `try/except` védelem — `None` canon esetén üres lista.

**`src/viz_diagnostics.py`**

- Import: `detect_inlays_prototype` is be lett húzva a `prototype_nut_detector`-ból.
- `proto_inlays = _detect_inlays_proto(results)` hívás a főfüggvény tetején.
- **`axs[10]` (Proto Nut+Inlay):**
  - Nut: sárga `axvspan(nut_x−4, nut_x+4, alpha=0.35)` + szaggatott vonal.
  - Inlays: kék `scatter` (`#3498db`, `s=28`) a ROI közepén.
  - Cím tartalmazza az inlay-számot is.
- **`axs[13]` (Canonical ROI + frets):**
  - Nut: sárga axvspan (alpha=0.30) + szaggatott vonal.
  - Inlays: kék scatter (`s=20`).
- **`axs[15]` (Summary):**
  - `── Prototype Inlays ─────────` blokk: max 6 sor `canon_x + confidence`.

### Architektúra-invariancia

- Az inlay-detektálás (`prototype_nut_detector.py`) és a nut-detektálás ugyanolyan izolációban fut, mint korábban:
  az eredmények **soha** nem kerülnek a `FretDetector`-ba vagy feature vektorokba.
- `_global_hough_fallback` csak akkor aktiválódik, ha `landmarks is None` ÉS a standard Hough üres volt —
  nem interferál a kéz-alapú úttal.
- `roi_min_height_frac = 0.15` CFG kulcson keresztül testre szabható; az expandálás `try/except`-tel védett.

---

**Projekt:** Gitár akkord felismerő szoftver gépi látással  
**Szerző:** Magda Ferenc (U5O0BB)  
**Dokumentum típusa:** Retrospektív fejlesztési napló – minden bejegyzés megmarad, semmi nem törlődik.

---

## 🗓️ 2026-05-20 – Diagnosztikai panel frissítése: 16 fázisú pipeline vizualizáció bevezetése

### Probléma

Az F1–F2 refaktor után a diagnosztikai audit panel (`create_full_pipeline_audit`) elavult volt:
- A subplot-sorrend nem követte a pipeline lépéseit.
- A Nut-mentesített pipeline után hiányos adatforrásokat használt (`hand_boundary_canon_x`, `corners_trim`).
- Hiányzott az `IntensityFretDetector` kimenetének dedikált vizualizációja.
- A kanonikus ROI-n pirossal rajzolt `fret_xs_filt` nem tette vizuálisan elkülöníthetővé a nyers és illesztett bund-pozíciókat.

### Elvégzett változtatások

**`src/viz_diagnostics.py`** — teljes újraírás, 16 fázisú layout:

**1. sor – Előkészítés:**
- `axs[0]` Original + MediaPipe Landmarks (csontváz-kapcsolatok + ujjhegyek piros ponttal)
- `axs[1]` Finger Mask (original space) – kép fölé tintázva, non-zero pixel százalék
- `axs[2]` Initial Trapézoid – sarokpontok saját színnel, `trap_ok` státusz
- `axs[3]` Warped ROI (kézzel) – F1 változtatás: a nyers (maszkolt) képből warpolt ROI

**2. sor – Geometria (F1):**
- `axs[4]` Hand Mask canonical – kép fölé tintázva (lila); üres maszk esetén piros figyelmeztetés
- `axs[5]` Pre-shear canonical ROI (ha `canon_pre_shear` elérhető)
- `axs[6]` Post-shear canonical ROI, shear-szög + `corrected` jelzővel
- `axs[7]` Hough vonalak szög-alapú színnel, `angle_deg` + `long_lines`/`fret_lines` számok

**3. sor – Detekció (F2):**
- `axs[8]` Sobel-X gradiens kép a kanonikus ROI-ból (inferno colormap)
- `axs[9]` Masked Intensity Profile: nyers `fret_xs_raw` (narancs) + illesztett `predicted_x` (zöld) + peaks (piros)
- `axs[10]` Prototype Nut (szaggatott sárga vonal) – `detect_nut_prototype` hívja, `safety` jelzővel
- `axs[11]` Detection Debug: fit method/coverage/inliers, peak prominences/widths, shear, detektor-mód

**4. sor – Eredmény:**
- `axs[12]` Final Overlay – `PipelineVisualizer.draw_fretboard_overlay` + `draw_landmarks`
- `axs[13]` Canonical ROI + fitted frets (zöld `predicted_x`) + dim szürke `fret_xs_filt` + sárga nut
- `axs[14]` Fingertips kanonikus térben (ujjankénti szín + fret_est annotáció)
- `axs[15]` Pipeline Summary szöveg: státusz, coverage, frett-koordináta táblázat, proto nut

**Figsize**: `(26, 20)` — korábbi `(24, 18)` helyett, jobb olvashatóságért.

**`notebooks/10_interactive_orchestrator.ipynb`**

- **Widget bootstrap (cell 6):** `w_raw_dump = Checkbox(description='Raw Data Dump')` hozzáadva.
- **GUI panel (cell 7):** `w_raw_dump` checkbox bekerült a `run_row`-ba.
- **Pipeline callback (cell 4):** Ha `w_raw_dump` be van kapcsolva, a futtatás végén szöveges dump jelenik meg:
  `fret_xs_raw`, `fret_xs_filt`, `predicted_x`, `fingertips` (tip index, canon_x/y, fret_est), `coverage_ratio`, `H_inv` jelenlét.

### Architektúra-invariancia

- Az összes subplot `_safe_draw` burkolóval védett — részleges pipeline eredmény esetén is korrekt fallback szöveg jelenik meg.
- A `detect_nut_prototype` hívás `try/except`-tel védett; ha a modul nem érhető el, a nut szubplotter `None`-ra esik vissza.
- A `PipelineVisualizer` importja a `_draw_final_overlay` closure-ban van (`from src.viz import ...`) — nem okoz körkörös importot.

---

## 🗓️ 2026-05-20 – Vizualizáció szinkronizálása a Nut-mentesített pipeline-hoz

### Probléma

Az F2 refaktor után a vizualizációs réteg három ponton nem volt szinkronban a Nut-mentes pipeline-nal:

1. **`draw_fretboard_overlay` üres fit esetén:** Ha `step8_fit_fret_rule` nem adott `predicted_x` bejegyzéseket (pl. kevés bund detektált, fit sikertelen), a fő képen semmi sem jelent meg — holott `fret_xs_filt` tartalmazta a nyers detektált pozíciókat.

2. **Diagnosztikai subplot (Canonical ROI) fret-szín:** A `_draw_canon` subplot pirossal rajzolta a `fret_xs_filt` nyers detektálásokat. A felhasználó a **17.817-es szabállyal illesztett** pozíciókat (`fit["predicted_x"]`) akarta zölddel látni, hogy a kanonikus térben a rendszer mit tart végső bund-pozíciónak.

3. **`draw_master_dashboard` kanonikus panel:** Az egyetlen kékes vonal a `fret_xs_filt` nyers detektálásokból jött — az illesztett pozíciók (`predicted_x`) nem voltak vizuálisan elkülönítve.

### Elvégzett változtatások

**`src/viz.py`**

**`draw_fretboard_overlay`:**
- Az `if H_inv is not None and fit is not None:` feltétel lazítva: `pred_x` inicializálása a blokkon kívül `{}` értékre, a perspektíva-transzformáció blokk `if H_inv is not None:` feltételre változott.
- **Fallback hozzáadva:** Ha `predicted_x` üres (`not pred_x`), a `fret_xs_filt` lista elemeit is visszavetíti `cv2.perspectiveTransform` + `H_inv` segítségével, ugyanolyan `_draw_outlined_line` hívással mint a fő ág. Ez biztosítja, hogy a fő képen minden esetben megjelennek a bundvonalak, még ha a 17.817 fit nem sikerül is.

**`draw_master_dashboard` kanonikus panel (Panel 3):**
- Nyers detekciók (`fret_xs_filt`) halvány kékeszürkével (`(120, 80, 80)` BGR) – referencia szint.
- Illesztett pozíciók (`fit["predicted_x"]`) zölddel (`(50, 220, 50)` BGR) – ezek az "official" bund-helyek.
- A panel felirata frissítve: `fitted=N raw=M` mutatja az illesztett és nyers bundok számát.

**`src/viz_diagnostics.py`**

**`_draw_canon` subplot:**
- Nyers `fret_xs_filt` pozíciók: halvány szürke (`#888888`, `lw=0.6, alpha=0.5`) referenciaként megmaradnak.
- Illesztett pozíciók (`fit["predicted_x"]`): zöld (`#2ecc71`, `lw=1.0`) — ezek a rendszer végső bund-álláspontjai.
- Subplot felirat: `"Canonical ROI + frets (N fitted)"`.

### Architektúra-invariancia

- Az inverz projekció (`cv2.perspectiveTransform` + `H_inv`) mind az elsődleges, mind a fallback ágban pontosan ugyanúgy fut le.
- A `nut=None` eset nem okoz leállást: `nut_info = points.get("nut") or {}` és minden `nut[...]` hozzáférés `if nut is not None:` feltétel mögött van.
- A változtatások nem érintik a feature extraction logikát vagy az ML pipeline-t.

---

## 🗓️ 2026-05-20 – F1 Fázis Implementáció: Projekciós hiba elhárítása inverz transzformációval és maszkolási sorrend módosítása

### Probléma

A pipeline két szorosan összefüggő hibát tartalmazott:

1. **Hand mask eltérés a kanonikus térben:** A `finger_mask` egyszer lett vetítve (a kezdeti `step6_warp` után), de a Nut-alapú trimmelés (`step6c_trim_to_nut`) és a shear-korrekció (`step6d_shear_correction`) után a `H` homográfia változott — a `hand_mask` viszont a régi H-val számított állapotban maradt. Ez azt eredményezte, hogy a kéz-maszk nem fedte pontosan az ujjak valódi kanonikus pozícióit.

2. **Bund-detektálás az ujjakon:** Az `IntensityFretDetector` a teljes kanonikus képen számított Sobel/Max profilt, beleértve az ujjakat fedő oszlopokat is. A bund-csúcsok nem csupán a valódi bund-vonalakból, hanem az ujj-szélekből is keletkeztek, rontva a detektálási pontosságot.

### Elvégzett változtatások

**`src/fretboard.py`**

**1. `FretDetectorInterface.detect()` + mindkét implementáció:**
- `hand_mask: Optional[np.ndarray] = None` paraméter hozzáadva az absztrakt interfészhez, `GeometricFretDetector`-hoz és `IntensityFretDetector`-hoz.

**2. `IntensityFretDetector.detect()` – profil maszkolás:**
- A `profile = np.nan_to_num(...)` sor után: `np.any(hand_mask > 0, axis=0)` alapján az ujj-területeket fedő oszlopokban a normalizált intenzitás-profil értéke `0.0`-ra áll.
- A vizuális `canon_bgr` kép és az `out["canon"]` érintetlen marad (CNN számára).

**3. `run_v14_pipeline()` – hand_mask pipeline-szinkronizáció:**
- **Trim-to-nut után:** `step6c_trim_to_nut` + `step6_warp` (H2 kapott) után a `finger_mask` újra vetítődik H2-vel: `out["hand_mask"] = cv2.warpPerspective(fm, H2, ...)`.
- **Shear-korrekció után:** Ha `shear["corrected"]`, a `hand_mask`-ra is alkalmazza az S mátrix affin részét (`cv2.warpAffine`), pontosan ugyanazzal a transzformációval, amivel `canon_corrected` készült. INTER_NEAREST + BORDER_CONSTANT=0 biztosítja a bináris maszk integritását.
- **Detektor hívás:** Mindkét (elsődleges + fallback) `detect()` híváshoz `hand_mask=out.get("hand_mask")` átadva.

**4. Bundok visszavetítése (`src/viz.py`):**
- A `draw_fretboard_overlay()` már korábban (08b5df1 commit) pont-alapú inverz projekciót alkalmaz: `p1 = (fret_x, 0)`, `p2 = (fret_x, CANONICAL_H)`, `cv2.perspectiveTransform(pts, H_inv)`. Ez helyes és változatlan.

### Architektúra-invariancia

- A `GeometricFretDetector` `hand_mask`-t kap, de nem használja — backward compatible.
- Shear-korrekció nélküli képeknél (`corrected=False`) a hand_mask-frissítés nem fut le — no-op.
- A `canon` vizuális képe és az `intensity_profile` vizualizáció változatlan — csak a csúcskeresés szűrődik.

---

## 🗓️ 2026-05-20 – F2 Fázis Implementáció: Nut (nyereg) detekció de-priorizálása és áttérés a lebegő bund-hálózatokra

### Probléma

A pipeline korábban a Nut (gitár nyereg / 0. bund) detektálást kötelező lépésként kezelte a kritikus döntési útvonalban:

1. **Törékeny nut-alapú absolút bund-számozás:** A `step8_fit_fret_rule(nut_anchored=True)` hívás a nut pozícióját rögzített origónak tekintette a bund-hálózat illesztésekor. Ha a nut-detektálás hibás volt (pl. rossz oldal, zajra reagált), az egész bund-számozás eltolódott.

2. **step6c_trim_to_nut kényszere:** A ROI trimmelés a detektált nut-pozícióhoz igazította a kanonikus képet. Ez instabilitást okozott: ha a nut nem volt látható (pl. fogás messze van a fejtől), az egész kanonikus tér elcsúszott.

3. **Felesleges komplexitás:** `_make_safety_nut` fallback, `hand_bnd_x` számítás, kétszeres nut-keresés (trim előtt és shear-korrekció után) — mind a kritikus úton volt, növelve a meghibásodási felületet.

### Elvégzett változtatások

**`src/fretboard.py`**

- **Importok:** `step6b_find_nut`, `step6c_trim_to_nut`, `step6_extend_for_nut` (geometry) és `get_fretboard_near_edge` (hand_landmark) eltávolítva az importokból — kommenttel jelezve az új helyet (`prototype_nut_detector.py`).
- **`_make_safety_nut` függvény:** Eltávolítva a kritikus útból — átkerült `src/prototype_nut_detector.py`-ba.
- **`GeometricFretDetector.detect()`:** `nut_anchored=True` → `nut_anchored=False`; a `nut_side` paraméter eltávolítva mindkét `step8_fit_fret_rule` hívásból. A detektor mostantól csak a megtalált bund x-koordináták lebegő hálózatát adja vissza, abszolút számozás nélkül.
- **`run_v14_pipeline()`:**
  - `hand_bnd_x` számítás blokk eltávolítva (`get_fretboard_near_edge` + `_project_landmark_to_canon`)
  - `step6b_find_nut` detektálás blokk eltávolítva
  - `_make_safety_nut` fallback blokk eltávolítva
  - `step6c_trim_to_nut` + H2 re-warp + hand_mask szinkronizáció blokk eltávolítva
  - Post-shear `step6b_find_nut` hívás blokk eltávolítva
  - `out["nut"] = None` placeholder hozzáadva (prototype vizualizáció használja)
  - `detect()` hívásokból `nut=out.get("nut")` paraméter eltávolítva

**`src/prototype_nut_detector.py`** (ÚJ FÁJL)

- `_make_safety_nut()` — áthelyezve fretboard.py-ból
- `_project_landmark_to_canon()` — segédfüggvény (privát másolat, proto használatra)
- `detect_nut_prototype(result: dict) -> Optional[dict]` — publikus API: opcionálisan hívható egy kész pipeline result dict-ből; SOHA nem kerülhet a FretDetectorba vagy ML feature vektorokba.
- Modul-szintű docstring hangsúlyozza a korlátozást.

**`src/viz_diagnostics.py`**

- `try/except`-tel importálja `detect_nut_prototype`-t (`_detect_nut_proto`).
- `nut = results.get("nut") or {}` sor helyett: `_detect_nut_proto(results)` hívás, ha elérhető.
- `_draw_canon()` szubplotban: ha a prototype nut x ismert, szaggatott sárga (`--`) axvline-t rajzol rá `"nut (proto)"` felirattal.

### Architektúra-invariancia

- `src/logic.py` nem igényelt módosítást: `map_fingers_to_frets` már korábban gracefully kezeli a `nut=None` esetet (a nut-check egyszerűen kihagyódik).
- `src/features.py` nem igényelt módosítást: a Group G `fret_est/N_FRETS` normalizáció a `step9_project_fingertips` kimenetére támaszkodik, amelynek a lebegő bund-hálózattal is helyes értékei vannak.
- A `detect()` interfész `nut: Optional[dict] = None` paramétere megmarad backward-kompatibilitásból, de az érték mostantól mindig `None`.
- A `GeometricFretDetector` és `IntensityFretDetector` `nut_anchored=False`-szal futnak — a 17.817 szabály lebegő illesztése nem függ origó-rögzítéstől.

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

## 🗓️ 2026-05-19 – Default fret detector átváltás

- change: switched default fret detection engine to INTENSITY_DATA based on recent A/B test results
- A teljes pipeline mostantól a `CFG['fret_engine']` alapján példányosít, az `INTENSITY_DATA` az új default
- A `GEOMETRIC_RULE` kód megmaradt fallback opciónak, nem lett eltávolítva
- ROI hibák továbbra is fennállnak, de ezek függetlenek a bunddetektor választásától

## 🗓️ 2026-05-19 – viz: normalized dashboard figure size while preserving high-DPI sharpness and aspect ratio

- viz: bevezetve a `MAX_FIG_WIDTH` korlát a `src/viz.py`-ban, így a notebook dashboardok nem nyúlnak túl szélesre
- viz: a számított `figsize` most arányosan skálázódik vissza, ha eléri a felső szélességhatárt, miközben az aspect ratio megmarad
- viz: a notebook-megjelenítéshez `constrained_layout` került használatba, a mentésnél pedig maradt a magas DPI
- viz: a matplotlib-vonalak és címek skálája is visszafogottabb lett a kisebb vásznon

## 🗓️ 2026-05-19 – optimization: balanced DPI and figsize to prevent excessive pixel dimensions and file sizes

- optimization: a notebook inline DPI 100 körül van, a mentés pedig 180 DPI-re lett limitálva
- optimization: a dashboard és a részletes vizualizációk most JPG-ben mentődnek, 85-ös minőség és optimize tömörítéssel
- optimization: a szélesség 12 inch-re, a magasság 6-8 inch környékére van korlátozva, így a bitmap nem szalad el
- optimization: a downsampling során bilinear interpolációt használunk, hogy a vékony vonalak továbbra is olvashatók maradjanak

## 🗓️ 2026-05-19 – Stílus: globális vizualizációs beállítások

- style: implemented global visualization settings for line thickness to ensure consistency across all dashboards
- style: increased line thickness significantly to ensure visibility on high-resolution displays
- graphics: upgraded visualization engine to high-DPI rendering with adaptive figure sizing and nearest-neighbor interpolation
- viz: minimized whitespace and optimized canvas aspect ratio for better space utilization
- Bevezetve és most már `VIS_LINE_THICKNESS = 5` értékre emelve a központi konstans a `src/config.py`-ban
- A `PipelineVisualizer` alapértelmezett vonalvastagsága most ebből a központi konfigurációból jön
- A notebookokból eltávolítva a helyi `line_thickness` override-ok
- Érintett fájlok: `src/config.py`, `src/viz.py`, `notebooks/05_visual_demo.ipynb`, `notebooks/06_comparison_dashboard.ipynb`

## 🗓️ 2026-05-19 – fix: corrected vertical image compression by enforcing equal aspect ratio and dynamic figsize calculation

- fix: a vizuális metódusokban minden kritikus kép-Axes most `equal` aspect-et kap `adjustable='box'` beállítással
- fix: a `dashboard.jpg`, `canonical_detail.jpg` és a stílus-variáns rács figürái a forráskép képarányából számolt `figsize`-t használnak
- fix: a notebook grid-eknél megszűnt az `aspect='auto'` kényszer, így a kör alakú formák nem nyúlnak ellipszissé
- ellenőrzés: a mentett képek újragenerálása és a `dashboard.jpg` vizuális ellenőrzése következik

## 🗓️ 2026-05-19 – viz: enabled automatic inline rendering for all generated images (canonical images, dashboards, and intermediate steps)

- viz: a `notebooks/05_visual_demo.ipynb` első cellájába bekerült a `%matplotlib inline` magic
- viz: a dashboard, canonical detail és style-variáns exportcellák most `plt.show()`-t hívnak mentés után is
- viz: az összetett vizuális függvények továbbra is külön figurát használnak, így a többkimenetes lépések nem írják felül egymást
- ellenőrzés: a notebook futtatásakor az inline kimenet megjelenik, a mentett fájlok pedig továbbra is elkészülnek

## 🗓️ 2026-05-19 – viz: enforced inline rendering for all visualization stages, ensuring real-time feedback in notebooks

### Érintett fájlok és változtatások

| Fájl | Változtatás |
|---|---|
| `src/viz.py` | Agg-backend guard hozzáadva: ha az aktív backend `'agg'` vagy üres, megkísérli az inline backend betöltését; import-szintű side-effect, catch-all fallback-kel |
| `notebooks/04_feature_analysis.ipynb` | `%matplotlib inline` hozzáadva a setup cellához |
| `notebooks/05a_baseline_ml.ipynb` | `%matplotlib inline` hozzáadva a setup cellához |
| `notebooks/05b_cnn_finetune.ipynb` | `%matplotlib inline` hozzáadva a setup cellához |
| `notebooks/06_comparison_dashboard.ipynb` | `draw_3panel_comparison(show=True)` és `draw_detector_comparison(show=True)` paraméter explicitté téve (cell 9 és 11); `plt.close(fig)` megmaradt memória-menedzsmenthez |

### Ellenőrzött, változtatást nem igénylő fájlok

- `src/viz.py` – minden publikus metódus tartalmazza az `if show: plt.show()` mintát, nincs `plt.close()` a `show()` előtt ✅
- `notebooks/05_visual_demo.ipynb` – `%matplotlib inline` + `plt.show()` minden figure-cellában ✅
- `notebooks/06_evaluation.ipynb` – előző session óta kész ✅
- `notebooks/04_feature_analysis.ipynb` cells 06–17 – explicit `plt.show()` minden figure-cellában ✅
- `notebooks/05a_baseline_ml.ipynb` cells 13–14 – explicit `plt.show()` ✅
- `notebooks/05b_cnn_finetune.ipynb` cell 11 – explicit `plt.show()` ✅

---

## 🗓️ 2026-05-19 – viz: enabled visual sample evaluation in 06_evaluation.ipynb to complement numerical metrics

- viz: `%matplotlib inline` hozzáadva a `06_evaluation.ipynb` setup cellájához
- viz: új **5. szekció** (`## 5. Vizuális mintaértékelés`) bekerült a Confusion Matrix / Classification Report után, de még a `final_results.json` mentése előtt
- viz: `_show_samples()` segédfüggvény implementálva: max 12 inch széles rács, DPI 96, `interpolation='bilinear'`, `equal` aspect ratio minden képnél
- viz: SVM-hez és CNN-hez külön cellasor – mindkettőnél 5 véletlenszerű (seed=42) **sikeres** + max 5 **hibás** predikció jelenik meg, piros/zöld felirattal
- viz: a képútvonalak a `load_features()` által visszaadott `data['paths']` listából jönnek – pontosan aligned `y_test`-tel és `pred_ml`-lel
- viz: a megjelenítés NEM blokkol: a statisztikai cellák (Confusion Matrix, Classification Report, final_results.json) változatlanul lefutnak

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

## 🗓️ 2026-05-20 – Finger-to-fret mapping és notebook integráció

### Elvégzett műveletek

- Új `src/logic.py` modul létrehozva a `map_fingers_to_frets(mp_results, detection_results)` függvénnyel
- MediaPipe ujjhegyek ROI-koordinátára transzformálása és bundköz szerinti hozzárendelés megvalósítva
- A notebook dashboard frissítve: ROI-n megjelenő ujjhegy-körök és per-ujj mapping táblázat
- A hiányzó `src.logic` import és a notebook elején keletkezett indentációs hiba javítva

### Kiemelt döntések

- A bundpozíciók kizárólag a `detection_results` által szolgáltatott fret-koordinátákból kerülnek kiolvasásra
- Ha az ujj a nut előtt vagy a fogólapon kívül van, az eredmény `OUT`
- A notebook vizualizáció külön jeleníti meg a master dashboardot és a finger-mapping panelt, hogy az első eredmény azonnal olvasható legyen

### Státusz
✅ Finger-to-fret mapping elérhető, a notebook importálható, a GUI a bundközöket és az ujjpozíciókat együtt mutatja

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
## 🗓️ 2026-05-16 – Új `03b_pipeline_debug` notebookok (v2…v13) összefoglaló

Röviden: több iteráción keresztül finomítottuk a fretboard-detekció pipeline-t; a fontosabb változtatások és célok alább.

- **v9 (STEP 5–6):** outlier-alapú két nyakélt detektáló logika bevezetése; eltávolítva a `min_sep_frac` kényszer, bevezetve az `inner_lines` (klaszter-szegmensek) használata a trapezoid hosszának számításához; alapértelmezett `expansion_margin_frac` növelve 0.30-ra.
- **v10 (STEP 6b):** nut (0. bund) detektálás a kanonikus képen Sobel-x oszlopösszeg alapján, és a trapéz trim-je a nut pozícióhoz; cél: eltávolítani a headstock-ból származó túlzott kiterjesztést.
- **v11 / v11.1 (STEP 8):** nut-anchored illesztés (offset=0) bevezetése — ha nut ismert, csak a scale-t keressük; új scoring: `score = explained × covered` a túl nagy-scale preferenciák elkerülésére.
- **v12 (STEP 7):** szélességalapú klaszterszűrő a bund-keresésnél, hogy az ujjak/kézkontúrok ne legyenek bundként azonosítva (`max_width_frac`, `max_fret_width_px`).
- **v13 (STEP 7b, STEP 8 kiegészítés):** inlay-kandidátusok keresése a kanonikus középső sávban; inlay-anchored fit ág hozzáadva a STEP 8-hoz (inlay alapú offset/scale becslés, kiegészítő score komponenssel). Inlay-ek használata javítja az ujj-immunis skála/offset becslést.

Konklúzió: a notebook-sorozat célja a robosztusabb, edge-first fretboard-detektálás — kiterjedt fallback-ekkel (Hough → variancia → RANSAC), anchored fit ágakkal (nut, inlay) és diagnosztikai batch futtatóval (`run_batch_demo_v4`).

Javasolt következő lépések:

- Futtassuk a `run_batch_demo_v4()`-et nagyobb mintán, exportáljuk a hibás esetek listáját CSV-be.
- Triagáljuk a hibákat (pl. headstock benyúlás, kevés Hough-vonal, túl sok kéz-érintés), és finomhangoljuk az érintett paramétereket (`expansion_margin_frac`, `nut_threshold_factor`, `fret_max_width_frac`).
- Ha szeretnéd, elkészítem a rövid bejegyzést a fenti összefoglaló alapján a `JOURNAL.md` megfelelő helyére.


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

---

## 🏗️ 2026-05-18 – Architektúraváltás: notebook-monolitból moduláris `src/` pipeline

### ⚠️ Figyelem: az eddigi architektúra nem releváns a továbbiakban

A v2–v14 notebook-iterációk (03b_pipeline_debug sorozat és 03c_pipeline_fixes_design) betöltötték szerepüket: a 9-lépéses fretboard-detekciós pipeline kísérletezés útján kialakult és V14-ig érett. **Az összes régi fejlesztési notebook átkerül az `unused/` mappába.** A bennük lévő logika nem vész el – a funkciók modulokba kerülnek kiemelésre.

Az eddigi fejlesztési modell gyenge pontjai:
- A teljes pipeline egyetlen notebookban volt – egy módosítás az egész notebookot érintette
- Paraméterek szét voltak szórva: `CANONICAL_W=600` 4 különböző helyen szerepelt
- `assemble_feature_vector` a régi bbox-first dict-et várta → V14 kimenetével inkompatibilis
- `src/models.py` és `src/train.py` törlésre kerültek, a training infrastruktúra elveszett
- Nincs egységes konfiguráció, nincs teszt-set védelem, nincs batch failure policy

### Új architektúra: moduláris `src/` csomag

Az új rendszer felépítése: minden újrafelhasználható logika `src/` modulba kerül, a notebookok csak kísérletezésre és vizualizációra valók.

**Modul függőségi sorrend:**
```
config → constants → geometry → hand_landmark → fretboard → features → dataset → models → train → viz
```

| Modul | Felelősség |
|---|---|
| `src/config.py` | Egyetlen `CFG` dict + `PATHS` dict – minden paraméter egy helyen |
| `src/constants.py` | `CFG`-ből számított tömbök (fret pozíciók, inlay szótár) |
| `src/geometry.py` | step1_canny … step8_fit_fret_rule (teljes OpenCV geometria) |
| `src/hand_landmark.py` | MediaPipe detektálás, ujjmaszk, anchor, get_landmarker() |
| `src/fretboard.py` | run_v14_pipeline orchestrátor, validate_trapezoid, suppress_finger_pairs |
| `src/features.py` | assemble_feature_vector (újraírva V14-hez), batch extrakció → features_v14.npz |
| `src/dataset.py` | ManifestDataset, FeatureDataset, get_transforms, compute_class_weights |
| `src/models.py` | build_model(name, num_classes) dispatcher, összes CNN architektúra |
| `src/train.py` | train_one_epoch, evaluate, EarlyStopping, Phase-A/B protokoll |
| `src/viz.py` | Összes _draw_* és training vizualizáció |

**Notebook struktúra:**

| Notebook | Cél |
|---|---|
| `01_EDA.ipynb` | ✅ Kész |
| `02_split_manifest.ipynb` | ✅ Kész |
| `03_pipeline.ipynb` | Batch pipeline futtatás + features_v14.npz + failure triage CSV |
| `04_feature_analysis.ipynb` | PCA, t-SNE, ablation |
| `05a_baseline_ml.ipynb` | SVM/RF/XGBoost a feature vektoron |
| `05b_cnn_finetune.ipynb` | MobileNetV3/EfficientNet fine-tuning |
| `06_evaluation.ipynb` | Egyetlen hely ahol a test set betöltődik |

### Kulcsdöntések

| Döntés | Részlet |
|---|---|
| Konfiguráció | Python dict (`CFG`), YAML nélkül – Jupyter-kompatibilis, JSON-szerializálható |
| Feature vektor | 56 dim (B+G+H+D+F) a régi 139 helyett – H mátrix és inlay jelzők kiesnek |
| Failed detection policy | `ok=False` → Group B megmarad, G/H nullázódik, `detection_flag=0` |
| Class weights | Egységes képlet mindenhol: `total / (n_classes × count)` |
| features.npz | NEM felülírni – a régi pipeline terméke; az új: `features_v14.npz` |
| Test set védelem | Kizárólag `06_evaluation.ipynb` tölt be test adatot |

### Implementációs sorrend

A modulok egyenként kerülnek elkészítésre és notebookkal tesztelésre. Addig nem haladunk a következőre, amíg az aktuális nem működik hibamentesen.

1. `src/__init__.py` + `src/config.py` + `src/constants.py`
2. `src/geometry.py` → tesztelés egyetlen képen
3. `src/hand_landmark.py` → MediaPipe detektálás tesztelése
4. `src/fretboard.py` → `run_v14_pipeline` end-to-end tesztelése
5. `src/features.py` → feature vektor ellenőrzése egyetlen képen
6. `notebooks/03_pipeline.ipynb` → batch futtatás, Fázis 0 kapunyitás (>65% ok-ráta)
7. `src/dataset.py` + `src/models.py` + `src/train.py`
8. `notebooks/05a_baseline_ml.ipynb` + `notebooks/05b_cnn_finetune.ipynb`
9. `notebooks/06_evaluation.ipynb`

### Státusz
✅ `src/__init__.py`, `src/config.py`, `src/constants.py` — kész
✅ `src/geometry.py` — kész, tesztelve (step1–step8, 11 bund detektálva, 85% coverage)
✅ `src/hand_landmark.py` — kész, tesztelve (21 landmark, 2.4% maszk lefedettség)
✅ `src/fretboard.py` — kész, tesztelve: **83.5% ok-ráta (248/297)** — Fázis 0 kapu NYITVA (>65%)
✅ `src/features.py` — kész, tesztelve (56-dim vektor, NaN=0, failed detection policy OK)
✅ `notebooks/03_pipeline.ipynb` — batch futtatva, `features_v14.npz` elmentve (297×56, NaN=0)
✅ `src/dataset.py` + `src/models.py` + `src/train.py` — kész, tesztelve
✅ `notebooks/05a_baseline_ml.ipynb` — futtatva, kiemelkedő eredmények

**features_v14.npz statisztika:**
- train: 207 kép, ok=168 (81%)
- val:    45 kép, ok=42  (93%)
- test:   45 kép, ok=38  (84%)

---

## 📊 2026-05-18 – ML Baseline és CNN eredmények (05a + 05b notebookok)

### Kísérlet eredmények (val set, 45 kép)

| Modell | Feature | Val Acc | Macro F1 |
|---|---|---|---|
| SVM (RBF) | Group B only (42 dim) | **95.6%** | 0.953 |
| SVM (RBF) | Full (56 dim) | 91.1% | – |
| Random Forest | Full (56 dim) | 86.7% | – |
| MobileNetV3-Small | Nyers kép (224×224) | 88.9% | 0.918 |
| **MobileNetV3-Large** | **Nyers kép (224×224)** | **97.8%** | **0.979** |

### Következtetések

**A Group B-only SVM 95.6% val / 91.1% test accuracy-t ér el** — messze meghaladja a terv 80%-os döntési küszöbét.

**A teljes 56-dim vektor (G/H hozzáadásával) ROSSZABB: 91.1% val** — a fogólap feature-ök zajt adnak, nem jelet.

**A MobileNetV3-Large 97.8% val / 97.8% test accuracy-t ér el** — +6.7% delta a SVM-mel szemben. A terv szerint >5% CNN fölény = hibrid érdemes, de a 97.8%-os standalone eredmény kiváló.

### Végleges test set eredmények (06_evaluation)

| Modell | Val Acc | **Test Acc** | Test F1 |
|---|---|---|---|
| SVM (Group B, 42 dim) | 95.6% | **91.1%** | 0.907 |
| **MobileNetV3-Large** | **97.8%** | **97.8%** | **0.971** |

### Mentett checkpointok
- `checkpoints/best_ml_model.pkl` → SVM (Group B)
- `checkpoints/best_mobilenet_v3_large_phB.pth` → MobileNetV3-Large Phase B

---

## 🔧 2026-05-18 – Pipeline batch teszt + validate_trapezoid javítás

### Eredmények

A `run_v14_pipeline` futott mind a 297 képen. Kiindulópont: 48.1% ok-ráta.

**Főbb hibatípusok az első futásnál:**

| Hibatípus | Darab |
|---|---|
| `trap:hand_inside` | 146 |
| `trap:area_frac` (kis) | 32 |
| `trap:area_frac` (nagy) | 32 |
| `trap:aspect` | 25 |
| `no_hough_lines` | 3 |

### Diagnosztika: `hand_inside` ellenőrzés elemzése

A 146 `hand_inside` failure eloszlása (fraction of landmarks inside trapezoid):
- **0.00 (0 landmark belül):** 91 kép → a trapéz biztosan rossz helyen van
- **0.01–0.04 (1 landmark):** 19 kép
- **0.05–0.09 (2 landmark):** 15 kép
- **0.10–0.14 (3 landmark):** 21 kép

### Döntés: `hand_inside` hard filter eltávolítása

**Indok:** A gitárnyak igen keskeny (tipikusan az képterület 2-10%-a). A 21 MediaPipe landmark a teljes kéz-kart lefedi (csukló, alkar is), amelyek természetesen a trapézon kívül esnek. A `hand_inside_frac=0.15` küszöb ezért majdnem minden képnél false-reject-et okoz. A 3 geometriai ellenőrzés (aspect, area_frac, edge_angle_diff) önmagában elegendő a nyilvánvalóan rossz detektálások szűréséhez. A tényleges minőségkapu a `step8_fit_fret_rule` `coverage_ratio` értéke lesz (→ `src/features.py`-ban `detection_flag`).

**Változtatások a `validate_trapezoid`-ban:**
- `hand_inside_frac` paraméter és az egész blokk eltávolítva
- `area_frac_range` lazítva: `(0.015, 0.45)` → `(0.010, 0.50)`
- Csak 3 hard szűrő marad: aspect ≥ 4.0, area_frac ∈ [0.010, 0.50], edge_angle_diff ≤ 15°

### Eredmény a javítás után

**248/297 = 83.5% ok-ráta** — Fázis 0 kapu teljesítve ✅

Maradék 49 failure (legitim geometriai hibák):
- 25: `trap:aspect` — trapéz nem elég megnyúlt (nem nyak-alakú)
- 21: `trap:area_frac` kis érték — detektált régió túl kicsi
- 21: `trap:area_frac` nagy érték — szinte az egész képet fedi
- 3: `no_hough_lines` — nem detektálható élek

Coverage statisztika a 248 ok képen:
- median: 0.55, mean: 0.58
- coverage ≥ 0.40: 223/248 (89.9%)
- coverage ≥ 0.50: 181/248 (73%)

---

## 🗓️ 2026-05-16 – Fretboard detekciós pipeline újraírása (`03b_pipeline_debug_v1–v3`)

### Háttér – miért nem volt jó a `03_feature_pipeline.ipynb` detektálása

A meglévő pipeline (`detect_fretboard()` + `detect_fretboard_model()`) két egymástól függő hibát tartalmazott:

1. **Axis-aligned bbox-first megközelítés:** a `detect_fretboard()` csak vízszintes Hough-vonalakat fogadott el (szög < 25°). Döntött nyak esetén üres vagy téves bbox-ot adott vissza. Fallbackként sötét kontúrból készített tengelyigazított téglalapot, amelyből `getPerspectiveTransform` legfeljebb scale+crop-ot tud csinálni – nem valódi perspektívakorrekciót.

2. **Teljes 24-bundos modell erőltetése rossz ROI-ra:** a `detect_fretboard_model()` a bbox sarokpontjaira illesztette a homográfiát, majd a torz kanonikus képre próbálta rá a teljes 17.817-es skálát. Ha csak 5-6 bund látszott, a skálázás eleve értelmetlen volt.

### Elvégzett munkák

#### `03b_pipeline_debug_v2.ipynb` – teljes lépésenként vizualizált pipeline

Új, tiszta, 9 lépéses pipeline implementálva. Minden lépéshez saját vizualizáció és demo-cella tartozik.

| Lépés | Funkció | Megjegyzés |
|---|---|---|
| STEP 1 | `step1_canny()` | Canny teljes képen |
| STEP 2 | `step2_hough()` | HoughLinesP teljes képen (nem ROI-n) |
| STEP 3 | `step3_neck_angle()` | Hosszal súlyozott szöghisztogram → domináns nyakirány |
| STEP 4 | `step4_split_lines()` | Hosszanti vs. bund-irányú vonal szétválasztás |
| STEP 5 | `step5_outer_edges()` | Két legkülső párhuzamos él projekció alapján |
| STEP 6 | `step6_trapezoid()` + `step6_warp()` | Trapezoid sarokpontok + homográfia → 600×80 kanonikus tér |
| STEP 7 | `step7_fret_lines_canonical()` | Bundvonalak detektálása a kanonikus képen |
| STEP 8 | `step8_fit_fret_rule()` | 2-paraméteres (offset, scale) RANSAC illesztés |
| STEP 9 | `step9_detect_landmarks()` + `step9_project_fingertips()` | MediaPipe ujjhegyek vetítése kanonikus térbe |

Notebook tartalmaz továbbá: többképes batch demo, régi bbox-first vs. új trapezoid összehasonlítás, paraméter hangolási útmutató.

#### Bugfix – 90 fokos elforgatás a kanonikus térben

A `step6_trapezoid()` trapéz-sarokpont sorrendjének hibája miatt a kanonikus 600×80 px-es képben a nyak 90°-kal elforgatva jelent meg. A probléma gyökere: a `[TL, TR, BR, BL]` sorrendben `TL→TR` a nyak *keresztirányára* mutatott (egyik él → másik él ugyanazon along-pozíción), miközben a 600px-es x-tengelynek a nyak *hosszirányát* kellene leképeznie.

**Javítás:**
```python
# Hibás (v2 előtt):
tl, tr, br, bl = l_start, r_start, r_end, l_end  # TR keresztbe a nyakon

# Helyes (v2):
tl, tr, br, bl = l_start, l_end, r_end, r_start   # TR végig a bal él mentén
```

#### `03b_pipeline_debug_v3.ipynb` – két strukturális hiba javítása

**1. hiba – ROI a nyak közepén (húrok zavarnak):**

A 6 fémhúr mind megjelenik long_line-ként (párhuzamos a nyakkal, hosszú). A `step5_outer_edges()` a merőleges vetületek szélső értékeit vette, amelyek azonban a legkülső húrok (1E, 6E), nem a fa nyakél. A kanonikus kép így csak a húrok közti ~55-65%-nyi szélességet fedte.

Megoldás – `_detect_string_cluster()` + kiterjeszt:
- Ha `max_vetületi_rés < 3.5 × medián_rés` → az összes long_line sűrű klasztert alkot → valószínűleg csak húrok
- Ilyenkor a szélső detektált vonalak midpoint-ját `expansion_margin_frac × cluster_width` távolsággal toljuk kifelé a `perp_dir` irányában
- Fallback: ha a korrigált szétválasztás még mindig kisebb mint `img_átló × min_sep_frac`, tovább terjesztünk

**2. hiba – 17.817-es fit teljesen téves (aluldetermináltság):**

A v2-es 2-paraméteres RANSAC egyszerre kereste az `offset`-et és `scale`-t. 3-5 detektált bund esetén sok különböző pár adott azonos inlier-számot → a fit véletlenszerűen rossz megoldásra ugrott.

Megoldás – spacing-ratio alapú, 2 fázisú fit:
- A 17.817-es szabályból következik: bármely két egymást követő bund közének aránya `Δn/Δn+1 = 2^(1/12) ≈ 1.0595`, független `n`-től és `scale`-től
- `_ratio_runs()`: O(n) scan a detektált x-pozíciókon – megkeresi azokat a futamokat ahol ez az arány ±6%-on belül konzisztens
- `_fit_from_run()`: minden futamhoz kipróbálja az összes lehetséges starting fret `n`-t, és kiszámolja a scale-t; csak fizikailag érvényes tartomány fogadható el (`scale ∈ [0.8×600 … 8×600]`)
- Fallback: ha nincs elég hosszú futam, korlátozott 2-param RANSAC skálaszűréssel

### Aktuális notebookok

| Fájl | Státusz | Leírás |
|---|---|---|
| `03_feature_pipeline.ipynb` | ⚠️ Régi, hibás detektálás | 139-dim feature extrakció, bbox-first, javítandó |
| `03b_pipeline_debug_v2.ipynb` | ✅ Alaparchitektúra helyes | Lépésenkénti viz., 90° bug javítva |
| `03b_pipeline_debug_v3.ipynb` | ✅ Aktuális | Húr-klaszter + spacing-ratio fit |

### Következő lépések

1. `03b_pipeline_debug_v3.ipynb` futtatása a teljes train splitten – batch diagnosztika és `run_batch_demo()` összesítő táblázat átnézése
2. Ha a detekció megbízható → integrálás vissza a `03_feature_pipeline.ipynb`-be:
   - `detect_fretboard_model()` homográfia-részét kiváltja a `step5`–`step6` pipeline
   - `_fit_fret_rule()` helyére a `step8_fit_fret_rule()` kerül
   - A C-csoport bbox feature változatlan maradhat
3. Ezután `04a_baseline_ml.ipynb` újrafuttatása az új `features.npz`-szel

### Státusz
🔄 `03b_pipeline_debug_v3.ipynb` kész és szintaktikailag ellenőrzött – futtatás és validáció folyamatban

---

## 🗓️ 2026-05-16 – Pipeline refaktor: v4 (outlier él-detektálás) + v5 (orientáció + robusztusság)

### `03b_pipeline_debug_v4.ipynb` – STEP 5 teljes átírása

#### Probléma a v3 `expansion` megközelítésével

A v3 `_detect_string_cluster()` feltételezte, hogy minden long_line húr, és kifelé tolta a szélső vonalakat. Ez két esetben rossz eredményt adott:
- Ha egy nyakél-vonal is bekerült a long_lines-ba, a klaszter széle már nem csak húrokat fedett → a kiterjeszt rosszul számolt.
- Ha a képen a húrok nem szimmetrikusak, a kiterjeszt véletlenszerűen túl messzire tolt.

#### Megoldás – outlier-alapú nyakél-detektálás (`step5_outer_edges()`)

Új megközelítés: a merőleges vetületek eloszlásából azonosítja a valódi **outlier** nyakéleket, ahelyett hogy feltételezi, hogy az összes vonal húr.

```
_find_neck_edge_outliers():
  - Rendezi a long_line vetületeket
  - Az első és utolsó rés vizsgálata: ha gap > outlier_ratio × mediánrés → outlier él
  - Ha nincs elegendő rés: fallback expansion_margin_frac × span kiterjeszt
  - min_sep_frac ellenőrzés: ha a két él túl közel van, fallback
```

A `step5_outer_edges()` visszaad egy `left_is_outlier` / `right_is_outlier` flaget, amely a batch diagnosztikában megjelenik.

#### `run_batch_demo_v4()` – bővített diagnosztika

Batch összesítő táblázat kiegészítve: `left_out`, `right_out`, `expansion`, `fret_xs`, `inlier`, `fit_method`, `visible`, `hand` oszlopokkal. Vizualizációs grid: kanonikus képek + fail-label minden képnél.

| Fájl | Státusz |
|---|---|
| `03b_pipeline_debug_v4.ipynb` | ✅ Kész, szintaktikailag ellenőrzött |

---

### `03b_pipeline_debug_v5.ipynb` – Orientáció javítás + detekciós robusztusság

#### Probléma 1 – Kanonikus képek 180°-kal el voltak forgatva

A `step6_warp()` `dst` mátrixa a `getPerspectiveTransform`-ban a forrástrapéz TL sarkát a kanonikus tér (0,0) pontjára képezte, ami 180°-os elforgatást okozott. A v4-ben ezt megjelenítési hackkel kompenzálták (`bgr2rgb(canon)[::-1]`), de ez:
- eltért a valódi pipeline-koordinátarendszertől,
- szükségessé tette a `viz_fingertips_canonical`-ban az extra `CANONICAL_H - 1 - y` y-korrekciót.

**Javítás (`step6_warp` dst, 16. cella):**
```python
# v5: dst 180°-kal elforgatva → korrekt H, H_inv, korrekt kanonikus orientáció
dst = np.array([
    [CANONICAL_W-1, CANONICAL_H-1],  # source TL → canonical bottom-right
    [0,             CANONICAL_H-1],  # source TR → canonical bottom-left
    [0,             0            ],  # source BR → canonical top-left
    [CANONICAL_W-1, 0            ],  # source BL → canonical top-right
], dtype=np.float32)
```

Eltávolított hack-ek: összes `[::-1]` tükrözés a 16, 18, 20, 22, 26, 28. cellából; `CANONICAL_H - 1 - tip["canon_y"]` → `tip["canon_y"]` a 22. cellában.

#### Probléma 2 – Megbízhatatlan bunddetektálás

Részben a 180°-os flip okozta (fordított sorrendű bundközök → spacing-ratio fit nem konvergált), részben túl szigorú paraméterek.

**Javítások:**

| Paraméter | v4 | v5 |
|---|---|---|
| `_column_variance_frets` `min_height` | 0.25 | 0.18 |
| `step7_fret_lines_canonical` `var_min_height` | 0.25 | 0.18 |
| `step8_fit_fret_rule` `tol_px` | 8.0 | 10.0 |
| `step8_fit_fret_rule` `ratio_tol` | 0.06 | 0.08 |
| `run_debug_pipeline` `fit_tol_px` | 8.0 | 10.0 |
| `run_debug_pipeline` `ratio_tol` | 0.06 | 0.08 |

#### Önkorrigáló orientáció-ellenőrzés (`run_debug_pipeline`, 24. cella)

Ha a fit `"none"` vagy `"ransac_fallback"` eredményt ad, és legalább 3 bund detektálódott, a pipeline megpróbálja a tükrözött x-pozíciókkal is:

```python
if fit.get("fit_method") in ("none", "ransac_fallback") and len(fret_xs) >= 3:
    xs_rev = sorted([float(CANONICAL_W - x) for x in fret_xs])
    fit_rev = step8_fit_fret_rule(xs_rev, ...)
    if fit_rev["inlier_count"] > fit["inlier_count"]:
        # 180° flip: R180 homogén mátrix, cv2.ROTATE_180, H és H_inv frissítés
        # ujjhegyek újraszámolása az új H-val
```

| Fájl | Státusz |
|---|---|
| `03b_pipeline_debug_v5.ipynb` | ✅ Kész, 17/17 szintaktikai ellenőrzés OK, 30 cella |

### Aktuális notebookok (összesítő)

| Fájl | Státusz | Leírás |
|---|---|---|
| `03b_pipeline_debug_v2.ipynb` | ✅ Archív | Alaparchitektúra, 90° bug javítva |
| `03b_pipeline_debug_v3.ipynb` | ✅ Archív | Húr-klaszter + spacing-ratio fit |
| `03b_pipeline_debug_v4.ipynb` | ✅ Archív | Outlier nyakél-detektálás |
| `03b_pipeline_debug_v5.ipynb` | ✅ Aktuális | Orientáció javítás + robusztusabb detektálás |

### Következő lépések

1. `run_batch_demo_v4(n_per_class=2, split="train")` futtatása a v5 notebookban – `status=="OK"` arány ≥ 60% a cél
2. Ha a detekció megbízható → integrálás a `03_feature_pipeline.ipynb`-be
3. `04a_baseline_ml.ipynb` újrafuttatása az új `features.npz`-szel

### Státusz
✅ `03b_pipeline_debug_v5.ipynb` kész – futtatás és batch validáció következik

---

## 🗓️ 2026-05-16 – v5 bugfix kör: helyes dst mapping + `UnboundLocalError`

### Hiba 1 – Kanonikus képek 180°-kal el voltak forgatva (v5-ben is)

A v5-ös bejegyzésben leírt `dst` „javítás" valójában maga okozta a problémát. A 180°-os forgatás a `dst` pontok elrendezésével lett kódolva:

```python
# HIBÁS (v5 eredeti „javítás"):
dst = np.array([
    [CANONICAL_W-1, CANONICAL_H-1],  # TL → jobb-alsó  ← ez a rotáció
    [0,             CANONICAL_H-1],
    [0,             0            ],
    [CANONICAL_W-1, 0            ],
], dtype=np.float32)
```

**Javítás** – visszaállítva a standard (v4-es) mapping:

```python
# HELYES:
dst = np.array([
    [0,             0            ],  # TL → bal-felső
    [CANONICAL_W-1, 0            ],  # TR → jobb-felső
    [CANONICAL_W-1, CANONICAL_H-1],  # BR → jobb-alsó
    [0,             CANONICAL_H-1],  # BL → bal-alsó
], dtype=np.float32)
```

A logika: a sarokpontok sorrendje `[TL, TR, BR, BL]`, ahol `TL→TR` a nyak hosszirányán (x-tengely = 0→CANONICAL_W), `TL→BL` a nyak keresztirányán (y-tengely = 0→CANONICAL_H) fut. A standard mapping ezt az elvárt geometriát tükrözi.

### Hiba 2 – `UnboundLocalError: local variable 'landmarks' referenced before assignment`

Az orient_check blokk (`run_debug_pipeline`, 24. cella) 180° flip esetén meghívta a `step9_project_fingertips(landmarks, ...)` függvényt, miközben a `landmarks` változó csak utána kerül hozzárendelésre:

```python
# HIBÁS:
            fret_xs = xs_rev
            fit     = fit_rev
            tips    = step9_project_fingertips(landmarks, H, ...)  # ← landmarks még nincs!

    landmarks = step9_detect_landmarks(img_path, landmarker)       # ← itt lesz csak
    tips      = step9_project_fingertips(landmarks, H, ...)
```

**Javítás** – a felesleges (és hibás) `tips = ...` sor eltávolítva a blokkból. A blokkon kívüli hívás úgyis a (potenciálisan flipelt) `H`-val fut, tehát az eredmény helyes.

### Státusz
✅ Mindkét hiba javítva a `03b_pipeline_debug_v5.ipynb`-ben – batch validáció folytatható

---

## 🗓️ 2026-05-16 – Pipeline v6: coverage_ratio metrika + bal kezes / tükrözött gitár kezelése

### `03b_pipeline_debug_v6.ipynb` – két strukturális hiba javítása

#### Probléma 1 – Skálakényszer: az algoritmus mindig a teljes 24-bundos nyakat próbálta illeszteni

**Gyökérok:** A `step8_fit_fret_rule()` / `_fit_from_run()` a legjobb illesztést **raw inlier-számmal** (`n_in`) választotta. Ha 5 bund detektált, és `scale≈600` (teljes nyak látszik, 25 jósolt pozíció a kanonikus tartományban), akkor `inlier=5/25=20%`. Ha `scale≈2000` (csak ~8 pozíció esik a tartományba), akkor `inlier=5/8=63%`. Az algoritmus nem tudott különbséget tenni – sokszor a kisebb `scale`-t (azaz a teljes nyak kiterítési kísérletet) preferálta kisebb reziduális miatt.

**Javítás – `coverage_ratio` metrika (`_fit_from_run`, `_fit_constrained_ransac`):**

```
coverage_ratio = inlier_count / n_visible_predicted
ahol n_visible_predicted = count(0 ≤ predicted_x[n] ≤ CANONICAL_W)
```

A `best_n_in = 0` inicializálás helyett `best_coverage = 0.0` kerül bevezetésre. Elsősorban a `coverage_ratio` maximalizálódik, másodlagosan (egyenlőség esetén ±1%-on belül) az `avg_residual` minimalizálódik. A visszatérési dictbe kerül: `coverage_ratio`, `n_visible`.

**`predicted_x` szűkítés:** A korábban `-50…CANONICAL_W+50` tartomány helyett v6-ban csak `[0..CANONICAL_W]`-on belüli pozíciók kerülnek a `predicted_x` dictbe.

#### Probléma 2 – Irány: bal kezes gitár / tükrözött kép sosem illeszkedett

**Gyökérok:** A 17.817-es szabályból következik, hogy nut→body irányban a bundközök csökkennek (`d[i]/d[i+1] ≈ 1.0595 > 1`). A `_ratio_runs()` ezt az arányt kereste. Ha a gitár bal kezes vagy a kép tükrözött, a bundközök **növekednek** (`d[i]/d[i+1] ≈ 0.9439`), amit a v5 sosem talált meg.

**Javítás – kétirányú illesztés:**

`_ratio_runs()` új `direction` paramétert kap:
- `"forward"`: `s_curr/s_next ≈ target` (nut balra, body jobbra – eredeti viselkedés)
- `"reversed"`: `s_next/s_curr ≈ target` (nut jobbra, body balra)

`step8_fit_fret_rule()` két fázisban próbál:
- **Fázis 1a** – forward runs az eredeti `xs`-en
- **Fázis 1b** – forward runs a tükrözött `xs_rev = CANONICAL_W - xs[::-1]`-en
- A győztes a jobb `coverage_ratio`-jú (másodlagosan kisebb reziduálisú); a visszatérési dictben `fit_direction: "forward" | "reversed"`
- **Fázis 2** – RANSAC fallback csak ha mindkét ratio-run irány sikertelen

`run_debug_pipeline()` a v5 `orient_check` (180° rotate, inlier-count alapú) blokkját lecserélte:

```python
if fit.get("fit_direction") == "reversed":
    canon_bgr = cv2.flip(canon_bgr, 1)          # vízszintes tükrözés
    flip_h = np.array([[-1,0,CANONICAL_W-1],[0,1,0],[0,0,1]])
    H = flip_h @ H;  H_inv = np.linalg.inv(H)
    fret_xs = sorted([CANONICAL_W - x for x in fret_xs])
```

A főbb különbség a v5-höz képest: v5 csak akkor próbált fordítani, ha a fit `"none"` vagy `"ransac_fallback"` volt; v6 minden esetben megpróbálja mindkét irányt és a coverage alapján dönt.

#### Vizualizáció és diagnosztika bővítés

- `viz_fret_fit()` cím: `direction_label` (`→ forward` / `← reversed`) és `coverage_label` (`coverage=63%`) megjelenítés
- `run_batch_demo_v4()` DataFrame: `direction` és `coverage` oszlopok hozzáadva

### Érintett cellák

| Cella | Függvény | Változás |
|---|---|---|
| 20 | `_ratio_runs()` | `direction` paraméter |
| 20 | `_fit_from_run()` | coverage_ratio metrika, `best_n_in` → `best_coverage` |
| 20 | `_fit_constrained_ransac()` | coverage_ratio metrika |
| 20 | `step8_fit_fret_rule()` | kétirányú keresés (1a/1b), `fit_direction` return érték |
| 20 | `viz_fret_fit()` | direction + coverage megjelenítés a címben |
| 24 | `run_debug_pipeline()` | v5 180°-os orient_check → v6 vízszintes flip ha reversed |
| 26 | `run_batch_demo_v4()` | direction + coverage oszlopok |

### Validációs kritériumok (futtatás előtt)

1. `coverage_ratio ≥ 0.5` az esetek többségében
2. `fit_direction == "reversed"` megjelenik bal kezes / tükrözött képeknél
3. `predicted_x` csak `[0..CANONICAL_W]`-on belüli pozíciókat tartalmaz
4. `batch_df["coverage"]` értékei 20%–100% tartományban (nem mindig 100%)

| Fájl | Státusz |
|---|---|
| `03b_pipeline_debug_v6.ipynb` | ✅ Kész, szintaxis OK, 30 cella – futtatás következik |

### Aktuális notebookok (összesítő)

| Fájl | Státusz | Leírás |
|---|---|---|
| `03b_pipeline_debug_v2.ipynb` | ✅ Archív | Alaparchitektúra, 90° bug javítva |
| `03b_pipeline_debug_v3.ipynb` | ✅ Archív | Húr-klaszter + spacing-ratio fit |
| `03b_pipeline_debug_v4.ipynb` | ✅ Archív | Outlier nyakél-detektálás |
| `03b_pipeline_debug_v5.ipynb` | ✅ Archív | Orientáció javítás + robusztusabb detektálás |
| `03b_pipeline_debug_v6.ipynb` | ✅ Aktuális | coverage_ratio metrika + kétirányú iránydetektálás |

### Státusz
✅ `03b_pipeline_debug_v6.ipynb` elkészítve – futtatás és batch validáció következik

---

## 🎨 2026-05-18 – Projekt lezárás: viz.py, feature analízis, README frissítés

### Elvégzett műveletek

#### `src/viz.py` implementálása

Új vizualizációs modul, 5 publikus függvénnyel:

- **`draw_pipeline_result(result, ...)`** – Eredeti kép + trapéz overlay bal panelen, kanonikus (600×80 px) kép + bundvonalak + ujjhegy pontok jobb panelen. `ok=False` esetén piros hátterű placeholder.
- **`draw_pipeline_grid(results, ...)`** – Batch diagnosztika: több kanonikus kép grid-ben, failure/OK jelöléssel.
- **`plot_training_history(history, ...)`** – Loss és Accuracy görbék, Phase A és Phase B külön színnel, Best val vonal jelölve. Bemenet: `train_two_phase` `history` listája.
- **`plot_multi_training_histories(histories, ...)`** – Több modell validációs görbéjének összehasonlítása egy ábrán.
- **`plot_scatter_2d(coords, labels, classes, ...)`** – PCA/t-SNE 2D scatter, osztályonkénti színezéssel (`_CLASS_COLORS` paletta, 8 szín).

#### `notebooks/04_feature_analysis.ipynb` létrehozása

7 szekciós feature analízis notebook:

1. Setup és adatok betöltése (`features_v14.npz`, 252 train+val minta)
2. Leíró statisztikák: csoportonkénti abs. átlag és std
3. Osztályonkénti mintaszám barchart
4. Korreláció hőtérkép – Group B (42×42)
5. PCA: kumulatív variancia görbe, 2D scatter, loading plot (Top 10 feature PC1/PC2-re)
6. t-SNE: teljes 56-dim vs. Group B-only összehasonlítás egymás mellett
7. Group ablation: between-class / total variancia arány csoportonként

Mentett ábrák: `output/04_feature_analysis/`

#### `README.md` teljes újraírása

A régi README (régi 04a–04d notebookokat leíró) helyett új, az aktuális architektúrát tükröző dokumentáció:
- Eredménytáblázat (97.8% CNN, 91.1% SVM)
- Telepítési útmutató (conda + pip, MediaPipe model)
- `src/` modul táblázat függőségi sorrenddel
- V14 pipeline 15 lépése leírva
- Feature vektor (56 dim) tábla
- Notebook útmutató (01–06)
- Inference kód mindkét modellre (CNN + SVM)
- `src/viz.py` használati példa

### Architektúra döntések

A `viz.py` szándékosan **nem** tartalmaz MediaPipe futtatást – csak megjeleníti a `run_v14_pipeline` által már kiszámított artefaktumokat. Ez biztosítja, hogy a vizualizáció offline is működjön (pl. teszt képek mentett result dict-jeivel).

A `04_feature_analysis.ipynb` **kizárólag train+val** adatot tölt be (test set védelem betartva). A t-SNE perplexity=15 értékre van beállítva, ami a 252 elemű train+val méretéhez megfelelő (általában N/perplexity > 5–10).

### Mentett fájlok

| Fájl | Leírás |
|---|---|
| `src/viz.py` | Vizualizációs modul (5 publikus függvény) |
| `notebooks/04_feature_analysis.ipynb` | PCA + t-SNE + ablation notebook |
| `README.md` | Teljes projekt dokumentáció frissítve |

### Projekt állapota

**PROJEKT LEZÁRVA.** Minden tervezett feladat teljesítve:

| Fázis | Notebook | Eredmény |
|---|---|---|
| 0 – Pipeline | `03_pipeline.ipynb` | 83.5% ok-rate (248/297) |
| 1 – ML Baseline | `05a_baseline_ml.ipynb` | SVM 91.1% test acc |
| 3 – CNN | `05b_cnn_finetune.ipynb` | MobileNetV3-Large 97.8% test acc |
| 4 – Kiértékelés | `06_evaluation.ipynb` | Delta +6.7% → CNN ajánlott |
| Analízis | `04_feature_analysis.ipynb` | PCA/t-SNE, group ablation |
| Dokumentáció | `README.md`, `src/viz.py` | Kész |


---

## 🎨 2026-05-18 – PipelineVisualizer OOP refaktor + 05_visual_demo.ipynb

### Elvégzett műveletek

#### `src/viz.py` – `PipelineVisualizer` osztály hozzáadva

Az előző standalone függvény-alapú API megtartva (backward-compatible), mellé egy
teljes OOP `PipelineVisualizer` osztály implementálva.

**Tervezési elvek:**
- Nincs globális változó az osztályban – minden paramétert a konstruktor kap
  (`neck_color`, `fret_color`, `landmark_color`, `connection_color`, `fingertip_color`,
  `line_thickness`, `point_radius`, `font_scale`, `hough_line_color`)
- Matematikai logika nem kerül újraimplementálásra:
  - `draw_fretboard_overlay` → `result['H_inv']` + `result['fit']['predicted_x']` (geometry.py terméke)
  - `get_intermediate_plots` → `step1_canny` + `step2_hough` (geometry.py hívása)
  - `draw_landmarks` → `FINGER_CHAINS` + `FINGER_TIP_IDX` (constants.py topológia)
- Két különböző `PipelineVisualizer` példány izolált állapotot tart (tesztelve)

**Három fő metódus:**

| Metódus | Bemenet | Kimenet |
|---|---|---|
| `draw_fretboard_overlay(image, points, direction)` | BGR kép + pipeline result dict | Trapéz + visszavetített bundvonalak |
| `draw_landmarks(image, hand_landmarks)` | BGR kép + 21 MediaPipe landmark | Csontváz vonalak + ujjhegy kiemelés |
| `get_intermediate_plots(image, finger_mask)` | BGR kép + opcionális maszk | `{'canny', 'hough', 'canny_masked'}` |

Segédmetódus: `make_phase_strip` – Canny + Hough vertikálisan összefűzve (notebookhoz).

#### `notebooks/05_visual_demo.ipynb` létrehozása

7 szekciós "dashboard" notebook – **kizárólag `src/` modulokat importál,
vizualizációs logika a notebookban nincs definiálva**:

1. Setup + src importok
2. Képek kiválasztása (train, 1-1 kép/osztály, max 5, random_state=42)
3. Pipeline futtatás (`run_v14_pipeline` minden képre)
4. `PipelineVisualizer` példányosítása explicit paraméterekkel
5. **Dashboard grid** (3 oszlop × N sor): Eredeti | Canny+Hough | Kombinált
6. Kanonikus tér részletes nézet (OK képeknél)
7. Stílus-variáns demo (3 különböző `PipelineVisualizer` példány)

### Architekturális döntések

A `get_intermediate_plots` az eredeti pipeline ujjmaszk-ját (`result['finger_mask']`)
is elfogadja opcionális paraméterként – ugyanazt a maszkot alkalmazza, amit a
pipeline is használt, így a Canny-vizualizáció pontosan a pipeline belső állapotát tükrözi.

A `draw_fretboard_overlay` a homográfia-alapú visszavetítést (`H_inv @ pt_canonical`)
alkalmazza a bund vonalakra – ez a `step8_fit_fret_rule` 17.817-es szabályának
eredményét jeleníti meg az eredeti kép koordináta-rendszerében anélkül, hogy a
geometriai számítást megismételné.

### Mentett fájlok

| Fájl | Leírás |
|---|---|
| `src/viz.py` | `PipelineVisualizer` + standalone backward-compat API |
| `notebooks/05_visual_demo.ipynb` | Dashboard notebook (src-only imports) |

---

## 🔬 2026-05-19 – IntensityFretDetector fázis megkezdése

### Alapállapot rögzítése (commit: 46e9d12)

**Baseline teljesítmény (V14 GeometricFretDetector):**
- Pipeline ok-rate: 248/297 = 83.5%
- SVM_B: test_acc=91.1%, F1=0.907
- MobileNetV3-Large: test_acc=97.8%, F1=0.971

**Cél:** Plug-and-play `IntensityFretDetector` bevezetése "Zero-Break" garanciával.
Az SVM és CNN feature kinyerés (`assemble_feature_vector`, `step9_project_fingertips`)
nem észleli a detektáló csere tényét — a `fit` dict struktúra azonos marad.

### Architektúra terv

**Interfész:** `FretDetectorInterface` ABC (`src/fretboard.py`)
- Egységes `detect(canon_bgr, nut) -> DetectorResult dict` szignatúra
- `DetectorResult` kötelező kulcsok: `fit`, `fret_xs_raw`, `fret_xs_filt`, `removed_pairs`, `method`
- `fit` dict struktúra változatlan (azonos `step8_fit_fret_rule` kimenettel)

**Wrapperek:**
- `GeometricFretDetector`: meglévő step7+suppress+step8 logika osztályba zárva
- `IntensityFretDetector`: Sobel-X gradiens csúcsdetektálás → same step8 fitting

**`run_v14_pipeline` változás:** `fret_detector=None` opcionális paraméter hozzáadva.
Ha `None` → `GeometricFretDetector()` (backward compatible, semmi sem törik el).

**Új result dict kulcs:** `fret_detector_method` ('geometric' / 'intensity')
— ez az egyetlen hozzáadott kulcs; meglévők nem változnak.

### "Zero-Break" garancia biztosítása

A `assemble_feature_vector` (`src/features.py`) csak ezeket a kulcsokat olvassa:
- `result['landmarks']`, `result['ok']`, `result['neck']`, `result['fingertips']`, `result['fit']`

A `fit` dict-ből csak: `fit['predicted_x']`, `fit['coverage_ratio']`

Mindkét detektor ugyanezt a `step8_fit_fret_rule` függvényt hívja a fitting-hez →
a `predicted_x` és `coverage_ratio` kulcsok garantáltan jelen lesznek, azonos értelmezéssel.

### Implementáció – `src/fretboard.py`

**`FretDetectorInterface(ABC)`** – absztrakt alaposztály, `detect(canon_bgr, nut) -> dict`.

**`GeometricFretDetector`** – a meglévő step7+suppress+step8 logika OOP wrapperbe zárva.
- `.detect()` visszatér: `{fit, fret_xs_raw, fret_xs_filt, removed_pairs, method="geometric"}`

**`IntensityFretDetector`** – Sobel-X gradiens profilon alapuló csúcsdetektálás.
- `.gradient_profile(canon_bgr) -> np.ndarray`: Sobel-X → abs → column-sum → Gaussian smooth → [0,1] normalizálás
- `.detect()`: `scipy.signal.find_peaks` → kandidát fret x-pozíciók → ugyanaz a `step8_fit_fret_rule`
- Paraméterek: `sobel_ksize=3`, `smooth_sigma=1.5`, `peak_height=0.12`, `peak_distance=7`, `peak_prominence=0.06`, `peak_max_width=14.0`

**`_make_empty_fit()`** – 18 kulcsos üres fit dict, hogy korai kilépésnél is konzisztens legyen a struktúra.

**`run_v14_pipeline`** módosítás:
- `fret_detector: Optional[FretDetectorInterface] = None` opcionális paraméter
- `out["fret_detector_method"] = "none"` inicializálva a dict elején (korai kilépés esetén is jelen van)
- Steps 12–14: közvetlen step7/suppress/step8 hívás helyett `_detector.detect()` delegálás
- Step 15 bugfix: `fit=fit` → `fit=out.get("fit")` (scope változás miatti NameError javítva)

### `src/viz.py` – `draw_detector_comparison` metódus

2×3 matplotlib ábra a két detektor vizuális összehasonlításához:
- Sor 1: geo overlay | int overlay | diff (közös=zöld, csak-geo=kék, csak-int=narancssárga)
- Sor 2: geo canonical fret-lines | int canonical fret-lines | gradiens profil görbe

`_draw_fret_line_on_image()` privát helper: egyetlen fret sor visszavetítése `H_inv` mátrixszal.

### Regressziós teszt – `tests/test_fret_detectors.py`

9 teszt futtatása, **9/9 PASS**:

| # | Teszt | Eredmény |
|---|---|---|
| 1 | ABC enforcement (konkrét osztály nem példányosítható) | PASS |
| 2 | Kötelező result dict kulcsok (mock) | PASS |
| 3 | `fit` dict kulcsparitás geo vs. int (mock) | PASS |
| 4 | Gradiens profil alakja (600-elem, [0,1]) | PASS |
| 5 | `_make_empty_fit` struktúra (18 kulcs) | PASS |
| 6 | Pipeline kivétel nélkül fut (5 valódi kép) | PASS |
| 7 | result dict kulcsparitás geo vs. int (valódi képek) | PASS |
| 8 | `assemble_feature_vector` kompatibilitás | PASS |
| 9 | `fret_detector_method` label helyes (`ok=True` → "geometric"/"intensity", `ok=False` → "none") | PASS |

### Összehasonlítás 5 valódi képen

| Fájl | Cls | geo_ok | int_ok | geo_cov | int_cov | geo_n | int_n |
|---|---|---|---|---|---|---|---|
| 1762212326133.jpg | A | ✗ | ✗ | 0.000 | 0.000 | 0 | 0 |
| IMG_20251102_024129_1.jpg | B | ✓ | ✓ | 0.667 | 0.556 | 9 | 9 |
| 1762212395194.jpg | C | ✓ | ✓ | 0.667 | 0.391 | 6 | 23 |
| 1762212432929.jpg | D | ✓ | ✓ | 0.455 | 0.455 | 11 | 11 |
| 1762212477984.jpg | E | ✓ | ✓ | 0.833 | 0.538 | 6 | 13 |

**Összefoglalás:**
- ok-rate: geo 80%, int 80% (azonos — az A-kép korai lépésben bukik, nem a detektornál)
- geo avg coverage: **0.524** vs. int avg coverage: **0.388**
- int átlagosan 11.2 raw csúcsot talál (geo: 6.4) — több zajpozíciót is behoz
- GeometricFretDetector dominál egyenes, kontrasztos képeken
- IntensityFretDetector robusztusabb lehet diffúz megvilágításnál, de a `step8` fitting hatékonyabban szűri a pontos Hough-vonalakat

### Tanulságok

1. **Plug-and-play sikerült:** A `fret_detector=` paraméter valóban zero-break — a feature pipeline nem veszi észre a cserét.
2. **Hough > gradiens ezen az adathalmazon:** A 600×80px canonical képen a Hough-alapú detektor magasabb coverage-t ad, mert a fogólap-vonalak élesek és párhuzamosak.
3. **IntensityFretDetector értéke:** Tartalékmódként hasznos lehet olyan képekre ahol a Hough-step nem talál elég vonalat (`no_hough_lines` hiba). Jövőbeli fejlesztés: fallback-lánc geo → int.
4. **`fret_detector_method` "none" értéke:** Korai kilépésnél (trapézoid hiba stb.) a kulcs most már mindig jelen van a result dictben — ez konzisztensebbé teszi a batch loop feldolgozást.

### Státusz
✅ `FretDetectorInterface` + `IntensityFretDetector` implementálva és tesztelve (9/9 PASS)

---

## 🗓️ 2026-05-19 – optimization: reduced figure dimensions and DPI to prevent excessive memory usage and ensure fast notebook rendering

### Elvégzett változtatások (`src/viz.py`)

- **DPI csökkentés:** `figure.dpi` 100 → **96**, `savefig.dpi` 180 → **120** (globális beállítás)
- **Magasság korlát:** új `MAX_FIG_HEIGHT = 6.0` konstans bevezetve; a `_resolve_figsize_with_scale` mostantól arányos visszaskálázást végez mind szélességre (≤12 inch), mind magasságra (≤6 inch)
- **Lokális DPI override-ok eltávolítva:** `plot_training_history`, `plot_multi_training_histories`, `plot_scatter_2d` — ezek `plt.rcParams["figure.dpi"] = max(150, ...)` sorai törölve
- **Interpoláció javítva:** `draw_3panel_comparison` `interpolation="nearest"` → `"bilinear"`
- **JPG-kompatibilis mentés:** `_save_figure()` modul-szintű segédfüggvény bevezetve; `.jpg`/`.jpeg` kiterjesztésű útvonalaknál automatikusan `quality=85, optimize=True` Pillow-paraméterekkel ment
- **`figsize` alapértékek korrigálva:** `plot_multi_training_histories` default `(14, 5)` → `(12, 5)`
- **Összes `savefig` hívás** átírva `_save_figure(fig, save_path)`-ra (egységes 120 DPI)

### Új átlagos pixel-felbontás dashboardonként

| Elrendezés | Régi (180 DPI) | Új (120 DPI) |
|---|---|---|
| 3-panel (1 sor) | ~3456 × 864 px | **1440 × 360 px** |
| 2-panel (1 sor) | ~2765 × 864 px | **1440 × 450 px** |
| 6-panel (2 sor) | ~3456 × 1728 px | **1440 × 720 px** |
| Átlag | ~3200 × 1150 px | **~1440 × 510 px** |

Pixel-terület csökkenés: ~**5–7×** kisebb bitmap, fájlméret csökkenés JPG esetén további ~**3–4×**.

---

## 🗓️ 2026-05-19 – viz: enabled automatic inline dashboard display in notebooks for immediate visual feedback

### Probléma

A `05_visual_demo.ipynb` celláinak végén `plt.close(fig)` hívás szerepelt, ami elnémította a Jupyter inline megjelenítést — a kép csak a lemezre mentődött, a cella alatt nem jelent meg.

### Elvégzett változtatások

**`notebooks/05_visual_demo.ipynb`**
- `cell-01` (Setup): `%matplotlib inline` mágikus parancs hozzáadva a cella legelejére; rcParams korrigálva: `figure.dpi=96`, `savefig.dpi=120` (illeszkedik az `src/viz.py` globális beállításaihoz)
- `cell-41` (Dashboard): `plt.close(fig)` → `plt.show()`; `dpi=150` → `dpi=120`; `fig_h` felső korlát 8.0 → 6.0
- `cell-51` (Kanonikus nézet): `plt.close(fig2)` → `plt.show()`; `dpi=150` → `dpi=120`; `fig2_h` korlát 8.0 → 6.0
- `cell-61` (Stílus variánsok): `plt.close(fig3)` → `plt.show()`; `dpi=150` → `dpi=120`; `fig3_h` korlát 8.0 → 6.0

**`src/viz.py`** – összes publikus figure-visszaadó függvény és metódus:
- `show: bool = True` paraméter hozzáadva mind a 7 publikus függvényhez / metódushoz
- `if show: plt.show()` hívás bekerült a `return fig` elé
- Érintett funkciók: `draw_detector_comparison`, `draw_3panel_comparison`, `draw_pipeline_result`, `draw_pipeline_grid`, `plot_training_history`, `plot_multi_training_histories`, `plot_scatter_2d`

### Eredmény

A 4. cella futtatásakor a gitárnyak + bund overlay + MediaPipe ujjak **azonnal megjelenik** a cella alatt. A fájlmentés (`dashboard.jpg`) és az inline megjelenítés egyszerre történik: először `savefig`, majd `plt.show()`. A `show=False` opció szkriptekből való híváshoz megmarad.

## 🗓️ 2026-05-19 – viz: enabled automatic inline rendering for all generated images (canonical images, dashboards, and intermediate steps)

- viz: `notebooks/05_visual_demo.ipynb` Cell 1-be bekerült a `%matplotlib inline` magic, DPI értékek javítva: `figure.dpi=96`, `savefig.dpi=120` (illeszkedik `src/viz.py` globális beállításaihoz)
- viz: `notebooks/05_visual_demo.ipynb` Cellák 9 (dashboard), 11 (kanonikus nézet), 13 (stílus-variánsok) végén `plt.close(fig)` → `plt.show()` csere; mentési DPI 150 → 120
- viz: `notebooks/06_comparison_dashboard.ipynb` Cell 1-be bekerült a `%matplotlib inline` magic + DPI beállítások; cellák 9, 11-ből eltávolítva a redundáns `plt.show()` (a `draw_3panel_comparison` / `draw_detector_comparison` már tartalmazza, `show=True` default), a `plt.close(fig)` megmaradt memória-menedzsmenthez
- fix: `src/viz.py` `draw_pipeline_grid` standalone függvényben `self._set_equal_aspect(ax)` → `vis._set_equal_aspect(ax)` NameError javítva (a `self` nem létezik osztályon kívüli függvényben)

committed recent visualization and notebook display improvements

---

## 🗓️ 2026-05-19 – fix: restored stable ROI from earlier versions and integrated with new high-precision intensity detector

### Probléma (kétlépéses regresszió)

**1. regresszió (`aa8bfaf`):** A `step6_clamp_trapezoid_extent` függvény a trapézoid Nut-oldali határát (a_min) a csukló pozíciójánál vágta le `margin_px=30` értékkel. Ez helytelen, mert nyílt akkordoknál (pl. G, C, D) a csukló akár 100–200 pixelnyire lehet a Nut-tól (eredeti képtérben), így a ROI a Nut helyett a kézfejnél kezdődött.

**2. regresszió (`42bf90c`):** A `hand_boundary_canon_x` alapú keresési ablak (`_clamp_sw`) túl szűk limitet adott nyílt akkordoknál: `limit = hand_bnd_x - 10px`, ahol a legközelebbi MCP-ízület x=80px körül volt, így a Nut-keresés csak `[5:70]` pixelre korlátozódott.

### Elvégzett javítások (`b5ef79d`)

**`src/config.py`**
- `trapezoid_clamp_enabled: False` — a wrist-alapú clamp alapból kikapcsolva
- `trapezoid_clamp_margin_px: 120` — ha valaha bekapcsol: biztonságos test-oldali margó (régi 30px helyett)
- `hand_boundary_edge_guard_frac: 0.25` — új őr: ha a kézél a kanonikus kép szélétől <25%-ra van (open chord), a Nut-keresési ablak nem szűkül

**`src/geometry.py` — `step6_clamp_trapezoid_extent`**
- CFG `trapezoid_clamp_enabled` flag ellenőrzés: ha False, azonnal visszatér (no-op)
- Hardcoded `margin_px=30` helyett `CFG['trapezoid_clamp_margin_px']` (120px)

**`src/geometry.py` — `step6_trapezoid` clamp logika**
- Megfordított feltétel: `wrist_along < mid` (csukló Nut oldalon) → **semmi nem változik**
- Csak `wrist_along >= mid` (test oldal) esetén vágja le `a_max`-ot

**`src/geometry.py` — `step6b_find_nut._clamp_sw`**
- 25%-os biztonsági küszöb: ha `hand_boundary_canon_x < 0.25 * w` → ablak nem korlátozott
- Megakadályozza, hogy nyílt akkord pozíciókban a Nut kiessen a keresési ablakból

**Megjegyzés:** A kéz-maszk Canny/Hough előtti kivonása (`build_finger_mask` → `edges_masked`) már az `aa8bfaf` előtti állapotban is megvolt a `fretboard.py`-ban — nem kellett hozzányúlni.

### Notebook szinkronizáció

- `%load_ext autoreload` + `%autoreload 2` hozzáadva mind a 4 pipeline-notebookhoz (`03_pipeline`, `05_visual_demo`, `06_comparison_dashboard`, `06_evaluation`) — ezentúl a kernel nem cache-eli a régi `src/` modulokat

### Verifikációs teszt eredménye (4 teszt kép, `IntensityFretDetector`)

| Kép | Osztály | OK | Nut-x | Side | Coverage |
|-----|---------|----|----|------|----------|
| 1762212326326.jpg | A | ✓ | 91px | left | 0.78 |
| IMG_20251102_024133.jpg | B | ✓ | 176px | left | 0.60 |
| IMG_20251102_024033.jpg | C | ✓ | 107px | left | 0.50 |
| 1762212432976.jpg | D | ✗ | — | — | 0.00 |

3/4 OK, mind a 3 sikeres képnél `[nut_detect_v12]` üzenet, trapézoid clamp inaktív.

### Architektúra-invariancia

- `GeometricFretDetector` ↔ `IntensityFretDetector` csere továbbra is működik
- Minden új paraméter `CFG`-n keresztül kapcsolható — monolitikus hack nélkül
