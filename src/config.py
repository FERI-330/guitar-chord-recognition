from __future__ import annotations
from pathlib import Path

_root = Path(__file__).resolve().parent.parent

# ── Nut width constraints ─────────────────────────────────────────────────
# A nut szélessége tipikusan kb. 1.5x egy bundvastagság, de maradjon kisebb,
# mint a kéz/ujjak vetített szélessége a kanonikus ROI-ban.
NUT_CONSTRAINTS: dict = {
    "min_width": 6.0,
    "max_width": 18.0,
}

# Visszafelé kompatibilitás régi kulcsnevekkel.
NUT_WIDTH_CONSTRAINTS: dict = {
    "min_px": NUT_CONSTRAINTS["min_width"],
    "max_px": NUT_CONSTRAINTS["max_width"],
}

# ── Fret detection engine defaults ────────────────────────────────────────
FRET_ENGINE = "INTENSITY_DATA"
FRET_ENGINE_FALLBACK = "GEOMETRIC_RULE"

# ── Vizuális alapértelmezések ─────────────────────────────────────────────
VIS_LINE_THICKNESS = 5

# ── Pipeline geometria ─────────────────────────────────────────────────────
CFG: dict = {
    # Vizuális megjelenítés
    "vis_line_thickness": VIS_LINE_THICKNESS,
    # Kanonikus tér mérete
    "canonical_w": 600,
    "canonical_h": 80,
    # Fretboard fizika
    "n_frets": 24,
    "fret_rule": 17.817,
    # Bunddetektáló motor alapértelmezés
    "fret_engine": FRET_ENGINE,
    # STEP 1 – Canny
    "canny_low": 25,
    "canny_high": 80,
    "canny_blur_ksize": 5,
    # STEP 2 – HoughLinesP
    "hough_threshold": 30,
    "hough_min_len_frac": 0.15,
    "hough_max_gap": 15,
    # Finger mask – forearm extension
    "forearm_extend_scale": 1.5,
    # STEP 5 – Outer edges
    "step5_angle_tol": 15,
    "step5_outlier_ratio": 2.5,
    "step5_expansion_margin_frac": 0.30,
    # STEP 6 – Trapézoid clamp (ROI stabilizálás)
    "trapezoid_clamp_enabled": False,    # wrist-alapú ROI clamp: KI (Nut regresszió ellen)
    "trapezoid_clamp_margin_px": 120,    # ha BE: test-oldali margó px (régi hardcoded: 30)
    # STEP 6 – Sanity checks (kezdeti, megengedőbb fázis)
    # sanity_min_aspect: 1.5 (volt: 2.5) — közeli képen a nyak rövidebbnek látszik arányban
    "sanity_min_aspect": 1.5,
    # max_frac: 0.90 (volt: 0.70) — közeli képen a nyak > 70%-ot is lefedhet
    "sanity_area_limits": {"min_frac": 0.004, "max_frac": 0.90},
    "sanity_max_edge_angle_diff_deg": 20.0,
    # trap_orient küszöb: Hough vonalak átlagos szöge, ami felett a "tall trap" elutasítódik.
    # 35° (volt: hardcoded 20°) — közeli, enyhén dőlt képek tolerálásához szükséges.
    "sanity_trap_orient_angle_thr": 35.0,
    # STEP 6b – Nut detektálás
    "nut_width_filter_enabled": True,   # FWHM-alapú nut vs. bund diszkrimináció
    "nut_constraints": NUT_CONSTRAINTS,
    "nut_width_constraints": NUT_WIDTH_CONSTRAINTS,
    "nut_min_width_px": 5.0,           # minimális FWHM px, ami nut-ra utal
    "nut_max_width_px": 24.0,          # maximális FWHM px (ujj kizárása); None = nincs felső korlát
    "nut_n_candidates": 5,              # top-N csúcs vizsgálata (argmax helyett)
    # ── Ujjbegy érintési pont korrekció ─────────────────────────────────────
    "touch_point_offset_ratio": 0.025,  # TIP → ujjbegy közepe felé tolt vertikális offset
    # ── Kéz-határvezérelt Nut-keresési sáv ──────────────────────────────────
    "hand_boundary_enabled": True,      # landmark-alapú keresési sáv korlátozás
    "nut_hand_margin_px": 10,           # biztonsági margó a kézél előtt (canon px)
    "hand_boundary_edge_guard_frac": 0.25,  # kézél < 25% képszéltől → ablak nem korlátoz
    # ── Nut-First: trapézoid kiterjesztése a kéz felé ────────────────────────
    "nut_extend_amin_enabled": True,    # a_min kiterjesztése landmark-alapon (Nut-First)
    "nut_fallback_extend_px": 80,       # statikus kiterjesztés ha nut nem detektálható
    "nut_extend_amin_margin_px": 120,   # extra margó a legközelebbi landmark mögé (px)
    # STEP 8 – Fret rule fitting
    "step8_tol_px": 12.0,
    "step8_ratio_tol": 0.10,
    "step8_scale_min_factor": 1.0,
    "step8_scale_max_factor": 8.0,
    # ── Fantom-bund szűrés (post-fit refine) ────────────────────────────────
    "fret_refine_enabled": True,        # kétlépéses illesztés a phantomok ellen
    "fret_refine_tol_px": 12.0,        # ±tol az előrejelzett pozícióhoz képest
    # ── Training ──────────────────────────────────────────────────────────
    "random_seed": 42,
    "img_size": 224,
    "batch_size": 16,
    "lr_phase_a": 1e-3,
    "lr_phase_b_head": 1e-4,
    "lr_phase_b_backbone": 1e-5,
    "epochs_a": 20,
    "epochs_b": 25,
    "patience": 7,
    "num_classes": 8,
}

# ── Előfeldolgozás konfiguráció ───────────────────────────────────────────
PREPROCESSING_CONFIG: dict = {
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    "clahe_enabled":        True,
    "clahe_clip_limit":     2.0,
    "clahe_tile_grid_size": (8, 8),
    # Gaussian Blur (pipeline-szintű pre-blur, elkülönül a Canny belső blur-jétől)
    "blur_enabled":         False,
    "blur_ksize":           3,
    # Normalizálás
    "normalize_enabled":    False,
    "normalize_method":     "minmax",   # "minmax" | "histogram_eq"
}

# ── Elérési utak ───────────────────────────────────────────────────────────
PATHS: dict = {
    "root": _root,
    "data": _root / "data",
    "manifest": _root / "data" / "split_manifest.csv",
    "features_v14": _root / "data" / "features" / "features_v14.npz",
    "model_dir": _root / "models",
    "checkpoint_dir": _root / "checkpoints",
    "output_dir": _root / "output",
}
