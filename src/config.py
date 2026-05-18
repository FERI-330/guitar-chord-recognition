from __future__ import annotations
from pathlib import Path

_root = Path(__file__).resolve().parent.parent

# ── Pipeline geometria ─────────────────────────────────────────────────────
CFG: dict = {
    # Kanonikus tér mérete
    "canonical_w": 600,
    "canonical_h": 80,
    # Fretboard fizika
    "n_frets": 24,
    "fret_rule": 17.817,
    # STEP 1 – Canny
    "canny_low": 25,
    "canny_high": 80,
    "canny_blur_ksize": 5,
    # STEP 2 – HoughLinesP
    "hough_threshold": 30,
    "hough_min_len_frac": 0.15,
    "hough_max_gap": 15,
    # STEP 5 – Outer edges
    "step5_angle_tol": 15,
    "step5_outlier_ratio": 2.5,
    "step5_expansion_margin_frac": 0.30,
    # STEP 8 – Fret rule fitting
    "step8_tol_px": 10.0,
    "step8_ratio_tol": 0.08,
    "step8_scale_min_factor": 1.0,
    "step8_scale_max_factor": 8.0,
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
