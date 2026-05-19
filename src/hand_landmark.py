"""
src/hand_landmark.py

MediaPipe kézdetektálás, landmark projekció, ujjmaszk-generálás, nyakszög horgonyzás.

Forrás: 03c_pipeline_fixes_design.ipynb (V14), cellák 17, 19, 21.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import CFG, PATHS
from src.constants import (
    CANONICAL_H, FINGER_CHAINS, FINGER_THICK_MULT, FINGER_TIP_IDX
)
from src.geometry import (
    _normalize_angle,
    step3_neck_angle,
    detect_guitar_orientation as _detect_guitar_orientation,
)


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe inicializálás
# ─────────────────────────────────────────────────────────────────────────────

def get_landmarker():
    """MediaPipe HandLandmarker létrehozása.

    Raises FileNotFoundError, ha models/hand_landmarker.task hiányzik.
    NEM próbál letölteni – offline Docker környezethez.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as e:
        raise ImportError("mediapipe csomag nem elérhető.") from e

    model_path = PATHS["model_dir"] / "hand_landmarker.task"
    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model nem található: {model_path}\n"
            f"Másold be a hand_landmarker.task fájlt a models/ mappába."
        )

    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9a – MediaPipe landmark detektálás
# ─────────────────────────────────────────────────────────────────────────────

def step9_detect_landmarks(img_path, landmarker) -> Optional[list[tuple]]:
    """MediaPipe kéz landmark detektálás.

    Visszaad: [(x_norm, y_norm, z_norm)]*21 vagy None ha nincs kéz / hiba.
    """
    if landmarker is None:
        return None
    try:
        import mediapipe as mp
        image = mp.Image.create_from_file(str(img_path))
        result = landmarker.detect(image)
        if not result.hand_landmarks:
            return None
        best = result.hand_landmarks[0]
        return [(float(lm.x), float(lm.y), float(lm.z)) for lm in best]
    except Exception as exc:
        print(f"  [mediapipe] hiba: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9b – Ujjhegy vetítés a kanonikus térbe
# ─────────────────────────────────────────────────────────────────────────────

def step9_project_fingertips(landmarks: Optional[list],
                             H: np.ndarray,
                             w_img: int,
                             h_img: int,
                             fit: Optional[dict] = None) -> list[dict]:
    """21 MediaPipe landmark → kanonikus tér vetítés.

    Visszaad: lista dict-ekből minden ujjhegyhez:
        tip_idx, canon_x, canon_y, string_norm (0–1), fret_est (int|None)
    """
    if landmarks is None or H is None:
        return []
    pred = fit.get("predicted_x", {}) if fit else {}
    results = []
    for tip_idx in FINGER_TIP_IDX:
        if tip_idx >= len(landmarks):
            continue
        xn, yn, _ = landmarks[tip_idx]
        px, py = xn * w_img, yn * h_img
        pt = np.array([px, py, 1.0])
        proj = H @ pt
        if abs(proj[2]) < 1e-9:
            continue
        cx = float(proj[0] / proj[2])
        cy = float(proj[1] / proj[2])
        str_norm = float(np.clip(cy / CANONICAL_H, 0.0, 1.0))
        fret_est = None
        if pred:
            fret_est = min(pred.keys(), key=lambda n: abs(pred[n] - cx))
        results.append({
            "tip_idx": tip_idx,
            "canon_x": cx,
            "canon_y": cy,
            "string_norm": str_norm,
            "fret_est": fret_est,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ujjmaszk generálás
# ─────────────────────────────────────────────────────────────────────────────

def build_finger_mask(img_shape: tuple,
                      landmarks: Optional[list],
                      finger_thick_scale: float = 1.0,
                      palm_pad_scale: float = 0.15,
                      dilate_px: int = 2) -> np.ndarray:
    """Bináris maszk (uint8, 0/255) az ujjak és tenyér területén 255.

    Vastagság STABIL referenciából számolódik (MCP–MCP medián távolság),
    így független az ujjszegmensek hosszától.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if landmarks is None:
        return mask

    pts_px = [(int(round(lx * w)), int(round(ly * h))) for (lx, ly, _) in landmarks]

    mcp_pts = np.array([pts_px[i] for i in (5, 9, 13, 17)], dtype=np.float64)
    mcp_dists = [np.linalg.norm(mcp_pts[i + 1] - mcp_pts[i]) for i in range(3)]
    hand_scale = float(np.median(mcp_dists)) if mcp_dists else 20.0
    if hand_scale < 1.0:
        hand_scale = 20.0

    for name, chain in FINGER_CHAINS.items():
        thick = max(int(round(hand_scale * FINGER_THICK_MULT[name] * finger_thick_scale)), 4)
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(mask, pts_px[a], pts_px[b], 255,
                     thickness=thick, lineType=cv2.LINE_AA)
        for idx in chain:
            cv2.circle(mask, pts_px[idx], radius=thick // 2 + 1,
                       color=255, thickness=-1)

    palm_idx = [0, 1, 5, 9, 13, 17]
    palm_pts = np.array([pts_px[i] for i in palm_idx], dtype=np.int32)
    hull = cv2.convexHull(palm_pts)
    cv2.fillConvexPoly(mask, hull, 255)

    palm_pad = max(int(round(hand_scale * palm_pad_scale)), 2)
    k_palm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (palm_pad * 2 + 1, palm_pad * 2 + 1))
    palm_only = np.zeros_like(mask)
    cv2.fillConvexPoly(palm_only, hull, 255)
    palm_only = cv2.dilate(palm_only, k_palm)
    mask = cv2.bitwise_or(mask, palm_only)

    # Alkar-kiterjesztés: a csukló irányában maszkoljuk a forearm-élek forrását
    forearm_extend = CFG.get("forearm_extend_scale", 1.5)
    if forearm_extend > 0:
        wrist_pt = np.array(pts_px[0], dtype=np.float64)
        mcp_center_pt = mcp_pts.mean(axis=0)
        forearm_dir = wrist_pt - mcp_center_pt
        fd_len = float(np.linalg.norm(forearm_dir))
        if fd_len > 1e-3:
            forearm_dir /= fd_len
            extend_px = int(round(hand_scale * forearm_extend))
            arm_end = (wrist_pt + forearm_dir * extend_px).astype(int)
            arm_thick = max(int(round(hand_scale * 0.8)), 6)
            cv2.line(mask, tuple(pts_px[0]), tuple(arm_end), 255,
                     thickness=arm_thick, lineType=cv2.LINE_AA)

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Fogólap-közeli kézél meghatározás
# ─────────────────────────────────────────────────────────────────────────────

def get_fretboard_near_edge(landmarks: Optional[list],
                            img_shape: tuple) -> Optional[tuple[float, float]]:
    """A kéz gitárnyak felőli szélének pixel-koordinátái.

    A fogón lévő kéznél az MCP-ízületek (5, 9, 13, 17) alkotják a gitár
    felőli vonalat; a legkülső MCP-pont határolja azt a területet, ahol
    a Nut még kereshető.  A csukló (0) iránya jelzi, melyik MCP-oldal
    a gitártest felőli (és melyik a fejléc/nut felőli).

    Visszaad: (x_px, y_px) pixel-koordináta, vagy None ha nincs landmark.
    """
    if landmarks is None or len(landmarks) < 18:
        return None
    h, w = img_shape[:2]
    pts = np.array([[lx * w, ly * h] for (lx, ly, _) in landmarks])

    wrist = pts[0]
    mcp_idxs = [5, 9, 13, 17]
    mcp_pts = pts[mcp_idxs]

    # A fogólap iránya: csuklótól az MCP-centroid felé
    mcp_center = mcp_pts.mean(axis=0)
    neck_dir = mcp_center - wrist
    neck_len = float(np.linalg.norm(neck_dir))
    if neck_len < 1e-3:
        return None
    neck_dir = neck_dir / neck_len

    # Minden MCP projekciója a nyak-irányra; a legkisebb projekciójú
    # az a pont, amelyik a legközelebb van a gitár fejléc/nut oldalához
    projs = [float(np.dot(pts[i] - wrist, neck_dir)) for i in mcp_idxs]
    nearest_mcp_idx = mcp_idxs[int(np.argmin(projs))]
    pt = pts[nearest_mcp_idx]
    return (float(pt[0]), float(pt[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Gitár-orientáció becslés MediaPipe landmarkokból
# ─────────────────────────────────────────────────────────────────────────────

def detect_guitar_orientation(landmarks: Optional[list],
                              img_shape: tuple) -> Optional[dict]:
    """Kompatibilitási wrapper a geometry.detect_guitar_orientation köré."""
    return _detect_guitar_orientation({"landmarks": landmarks, "img_shape": img_shape})


# ─────────────────────────────────────────────────────────────────────────────
# Nyakirány horgonyzás MediaPipe landmark alapján
# ─────────────────────────────────────────────────────────────────────────────

def anchor_neck_angle(landmarks: Optional[list], img_shape: tuple) -> Optional[dict]:
    """Kéztengely → várható nyak-irány és palm centroid.

    Visszaad: {'expected_angle_deg', 'palm_center_px', 'wrist_px',
               'mcp_center_px', 'hand_angle_deg'} vagy None.
    """
    if landmarks is None:
        return None
    h, w = img_shape[:2]
    pts = np.array([[lx * w, ly * h] for (lx, ly, _) in landmarks])

    wrist = pts[0]
    mcp_idxs = [5, 9, 13, 17]
    mcp_pts = pts[mcp_idxs]
    mcp_center = mcp_pts.mean(axis=0)
    palm_center = (wrist + mcp_center) / 2.0

    hand_vec = mcp_center - wrist
    hand_angle = float(np.degrees(np.arctan2(hand_vec[1], hand_vec[0])))
    expected = _normalize_angle(hand_angle + 90.0)
    return {
        "expected_angle_deg": expected,
        "palm_center_px": palm_center,
        "wrist_px": wrist,
        "mcp_center_px": mcp_center,
        "hand_angle_deg": _normalize_angle(hand_angle),
    }


def step3_neck_angle_anchored(lines: list,
                              anchor: Optional[dict] = None,
                              window_deg: float = 30.0) -> dict:
    """step3_neck_angle horgonyzott változata: csak az anchor ± window_deg sávban keres.

    Ha anchor=None → sima step3_neck_angle-ra esik vissza.
    """
    base = step3_neck_angle(lines)
    if anchor is None or base["hist"].size == 0:
        return base
    centers = 0.5 * (base["bin_edges"][:-1] + base["bin_edges"][1:])
    target_base = anchor["expected_angle_deg"]
    best_peak, best_angle = -1.0, base["angle_deg"]
    best_target = target_base

    for target in (target_base, _normalize_angle(target_base + 90.0)):
        diff = np.abs(((centers - target + 90.0) % 180.0) - 90.0)
        mask = diff <= window_deg
        if not mask.any():
            continue
        hist_masked = base["hist"].copy()
        hist_masked[~mask] = 0.0
        if hist_masked.sum() <= 0:
            continue
        peak = int(np.argmax(hist_masked))
        peak_val = float(hist_masked[peak])
        if peak_val > best_peak:
            best_peak = peak_val
            best_target = target
            lo = max(0, peak - 1)
            hi = min(len(hist_masked) - 1, peak + 1)
            cc = centers[lo:hi + 1]
            ww = hist_masked[lo:hi + 1]
            best_angle = (float(np.dot(ww, cc) / ww.sum())
                          if ww.sum() > 0 else float(centers[peak]))

    if best_peak < 0:
        return base
    return {**base, "angle_deg": best_angle, "anchored": True,
            "anchor_angle_deg": best_target,
            "anchor_alt_used": (best_target != target_base)}
