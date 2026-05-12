## 2026-05-12 – EDA fázis indulása

- Létrehozva: `notebooks/01_EDA.ipynb` az adathalmaz feltérképezéséhez.
- Cél: osztályeloszlások, képminőség és fájlstruktúra részletes vizsgálata (`data/all`, `data/training`, `data/test`).
- Következő lépés: notebook futtatása lokálisan, döntési pontok rögzítése (képméret, színcsatornák, outlier kezelés).

---
## 2026-05-12 – 01_EDA.ipynb továbbfejlesztése (v2)

### Változások az előző verzióhoz képest
- **+** Tartalomjegyzék horgonyokkal (1–10. szekció)
- **+** `build_inventory()` → pandas DataFrame-alapú adatleltár mindhárom splitre, pivot összefoglalóval
- **+** Imbalance ratio kiszámítása (automatikus figyelmeztetés 2×, 3× felett)
- **+** Pixelintenzitás / RGB csatorna-eloszlás vizualizáció + saját mean/std értékek kiszámítása
- **+** MD5 alapú duplikátum-keresés (train vs. test leakage ellenőrzés)
- **+** Sérült képek `PIL.verify()` alapú szűrése
- **+** Fájlnév-forrás analízis (`IMG_` kamera vs. timestamp-appok) – osztályonkénti bontásban
- **+** Osztályonkénti kép-grid (n×4 subplot, balra az osztálycímke)
- **+** Train/test konzisztencia-ellenőrzés + osztályonkénti arány táblázat
- **+** Összefoglaló döntési táblázat (10 pont) a következő preprocessing fázishoz
- **Módosított:** `collect_image_stats()` – scatter plot (w vs. h) + pixel mód megoszlás hozzáadva

### Státusz
EDA notebook v2 kész, futtatható. Következő döntés: normalizálási értékek véglegesítése (saját vs. ImageNet).

---
