"""
src/roi.py

ROI (Region of Interest) kinyerési stratégiák.

Három megközelítést kínál egy egységes dict-kimenettel, amelyet a notebook
vizualizációk és az összehasonlító dashboard használ.

  roi_old_stable()         – Hough-alapú, MediaPipe nélkül (baseline)
  roi_mediapipe_guided()   – Teljes V14 pipeline (anchor + finger mask + nut-trim)
  roi_intensity_based()    – Sobel sor-projektálás, nincs perspektívakorrekció

Forrás: 08_detection_sandbox.ipynb Cell 3 (refaktorálva).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.config import CFG, PREPROCESSING_CONFIG
from src.constants import CANONICAL_W, CANONICAL_H
from src.geometry import (
    step1_canny, step2_hough, step3_neck_angle, step4_split_lines,
    step5_outer_edges, step6_trapezoid, step6_warp,
    bgr2rgb,
)


def roi_old_stable(img_bgr: np.ndarray) -> Optional[dict]:
    """Geometrikus Hough-alapú ROI, MediaPipe nélkül.

    Pontosan a V14 előtti megközelítés: Canny → Hough → neck_angle
    (anchor nélkül) → split → outer_edges → trapézoid → warp.
    Nincs finger-maszk, nincs anchor-korrekció, nincs nut-trim.

    Visszaad: dict pipeline-artefaktumokkal, 'ok' kulccsal, vagy None hiba esetén.
    """
    edges = step1_canny(img_bgr)
    lines = step2_hough(img_bgr, edges)
    if not lines:
        return None
    neck  = step3_neck_angle(lines)
    split = step4_split_lines(lines, neck["angle_deg"])
    if not split["long_lines"]:
        return None
    edge_info = step5_outer_edges(split["long_lines"], neck["angle_deg"])
    if edge_info is None:
        return None
    trap = step6_trapezoid(img_bgr, edge_info)
    if trap is None:
        return None
    H, H_inv, canon = step6_warp(img_bgr, trap["corners_px"])
    return {
        "label":  "Old Stable",
        "img":    img_bgr,
        "edges":  edges,
        "trap":   trap,
        "canon":  canon,
        "H":      H,
        "H_inv":  H_inv,
        "ok":     True,
    }


def roi_mediapipe_guided(img_path,
                          preprocessor=None,
                          landmarker=None) -> dict:
    """Teljes V14 pipeline (anchor + finger mask + nut-trim + shear correction).

    Visszaad: run_v14_pipeline kimenete 'label' kulccsal kiegészítve.
    """
    from src.fretboard import run_v14_pipeline
    entry  = {"path": img_path, "class": "?"}
    result = run_v14_pipeline(entry, landmarker=landmarker, preprocessor=preprocessor)
    result["label"] = "MediaPipe Guided"
    return result


def roi_intensity_based(img_bgr: np.ndarray,
                         row_frac: float = 0.30,
                         smooth_sigma: float = 5.0) -> Optional[dict]:
    """Sobel sor-projektálás alapú ROI, nincs Hough, nincs perspektívakorrekció.

    A vízszintes gradiens-energia sor-összege azonosítja a fretboard sávot.
    Kimenet: téglalap-crop, 600×80-ra nyújtva (nem warpolt perspektíva).

    Visszaad: dict vagy None ha a sáv nem azonosítható.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    row_energy = sx.mean(axis=1)
    row_smooth = gaussian_filter1d(row_energy, sigma=smooth_sigma)

    threshold   = row_smooth.max() * row_frac
    active_rows = np.where(row_smooth > threshold)[0]
    if len(active_rows) < 3:
        return None

    band_h = int(active_rows[-1]) - int(active_rows[0])
    pad    = max(int(band_h * 0.15), 5)
    y1     = max(0,     int(active_rows[0])  - pad)
    y2     = min(h - 1, int(active_rows[-1]) + pad)
    crop   = img_bgr[y1:y2, :].copy()
    canon  = cv2.resize(crop, (CANONICAL_W, CANONICAL_H))

    return {
        "label":      "Intensity-based",
        "img":        img_bgr,
        "row_energy": row_smooth,
        "y1":         y1,
        "y2":         y2,
        "crop":       crop,
        "canon":      canon,
        "ok":         True,
    }


def run_all_strategies(img_path,
                        preprocessor=None,
                        landmarker=None) -> tuple[Optional[dict], dict, Optional[dict]]:
    """Futtatja mindhárom ROI stratégiát egyetlen képre.

    Visszaad: (r_old, r_mp, r_int) tuple.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Kép nem olvasható: {img_path}")
    r_old = roi_old_stable(img_bgr)
    r_mp  = roi_mediapipe_guided(img_path, preprocessor=preprocessor, landmarker=landmarker)
    r_int = roi_intensity_based(img_bgr)
    return r_old, r_mp, r_int
