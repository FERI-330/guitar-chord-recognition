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
