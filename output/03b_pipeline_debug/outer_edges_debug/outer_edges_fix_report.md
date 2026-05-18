# outer_edges_fix_report.md

## Összefoglaló

| Metrika | Érték |
|---------|-------|
| Összes kép | 10 |
| Sikeres | 10 |
| Sikertelen | 0 |
| Mindkét outlier | 1 |
| Egy outlier | 3 |
| Mindkét bővített | 6 |
| DBSCAN fallback | 1 |
| sep min/átl/max | 0 / 382 / 751 px |

## Elvégzett változtatások

- `step5_outer_edges`: `_debug` dict a visszatérési értékben (`proj_vals`, `gaps`, `final_sep`, `fallback_used`, stb.)
- `_dbscan_1d`: pure-numpy 1D DBSCAN fallback (`USE_DBSCAN_FALLBACK = True` kapcsoló)
- `save_edge_debug_json`, `plot_proj_scatter`, `plot_proj_hist`, `plot_edge_overlay`: debug mentők
- 7 szintetikus unit teszt (`tests_results.txt`)

## Részletes eredmények

- seed=42 | F | bal=outlier jobb=bővített | sep=401px | fb=False
- seed=43 | B | bal=outlier jobb=bővített | sep=440px | fb=False
- seed=564210 | B | bal=bővített jobb=bővített | sep=341px | fb=False
- seed=1686 | F | bal=bővített jobb=bővített | sep=751px | fb=False
- seed=51770 | E | bal=bővített jobb=bővített | sep=378px | fb=False
- seed=923905 | F | bal=bővített jobb=outlier | sep=452px | fb=False
- seed=54898 | C | bal=bővített jobb=bővített | sep=185px | fb=False
- seed=967762 | B | bal=bővített jobb=bővített | sep=514px | fb=False
- seed=940983 | D | bal=bővített jobb=bővített | sep=0px | fb=True
- seed=329223 | G | bal=outlier jobb=outlier | sep=357px | fb=False

## Következő lépések

1. Ellenőrizd az overlay képeket azoknál, ahol `fallback_used=True`.
2. Ha sok a bővített eset: csökkentsd `outlier_ratio`-t 2.5→2.0-ra.
3. Ha `final_sep` rendszeresen nagy: csökkentsd `expansion_margin_frac` 0.30→0.20-ra.
4. Ha DBSCAN eps nem megfelelő: próbáld `img_span * 0.015` értékkel.