"""
src/fretboard.py

Teljes run_v14_pipeline orchestrátor + validálók + suppress.

Forrás: 03c_pipeline_fixes_design.ipynb (V14), cellák 23, 25, 27.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.config import CFG
from src.constants import CANONICAL_W, CANONICAL_H
from src.geometry import (
    bgr2rgb, load_image_bgr,
    step1_canny, step2_hough, step3_neck_angle, step4_split_lines,
    step5_outer_edges, step6_trapezoid, step6_warp,
    step6b_find_nut, step6c_trim_to_nut,
    step7_fret_lines_canonical, step8_fit_fret_rule,
)
from src.hand_landmark import (
    get_landmarker,
    step9_detect_landmarks, step9_project_fingertips,
    build_finger_mask, anchor_neck_angle, step3_neck_angle_anchored,
)

# Modul-szintű lazy singleton a landmarker-hez
_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        _landmarker = get_landmarker()
    return _landmarker


# ─────────────────────────────────────────────────────────────────────────────
# Trapézoid validálás
# ─────────────────────────────────────────────────────────────────────────────

def validate_trapezoid(corners: np.ndarray,
                       img_shape: tuple,
                       landmarks: Optional[list] = None,
                       min_aspect: float = 4.0,
                       area_frac_range: tuple = (0.010, 0.50),
                       max_edge_angle_diff_deg: float = 15.0) -> tuple[bool, list]:
    """Trapéz épelméjűségi szanitás (3 geometriai szűrő).

    A `hand_inside` ellenőrzés el lett távolítva: a gitárnyak keskenysége miatt
    a csukló/kar landmark-ok természetesen a trapézon kívül esnek, így a küszöb
    szinte minden képet elvett (148/297 false-reject). A 3 geometriai szűrő
    (aspect, area_frac, edge_angle_diff) elegendő a nyilvánvalóan rossz
    detektálások szűréséhez; a fret-fit quality (coverage_ratio) a tényleges
    minőségkapuként szolgál a features.py-ban.

    Visszaad: (ok: bool, reasons: list[str])
    """
    h, w = img_shape[:2]
    img_area = float(h * w)
    reasons = []

    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    tl, tr, br, bl = corners

    edge_lens = sorted([
        float(np.linalg.norm(tr - tl)),
        float(np.linalg.norm(br - tr)),
        float(np.linalg.norm(bl - br)),
        float(np.linalg.norm(tl - bl)),
    ])
    short_axis = (edge_lens[0] + edge_lens[1]) / 2.0
    long_axis = (edge_lens[2] + edge_lens[3]) / 2.0
    aspect = long_axis / max(short_axis, 1e-3)
    if aspect < min_aspect:
        reasons.append(f"aspect {aspect:.2f}<{min_aspect}")

    area = 0.5 * abs(
        np.dot(corners[:, 0], np.roll(corners[:, 1], -1)) -
        np.dot(corners[:, 1], np.roll(corners[:, 0], -1))
    )
    frac = area / img_area
    if not (area_frac_range[0] <= frac <= area_frac_range[1]):
        reasons.append(f"area_frac {frac:.3f} ∉ {area_frac_range}")

    edge_vecs = [tr - tl, br - tr, bl - br, tl - bl]
    edge_pairs = sorted(enumerate(edge_vecs),
                        key=lambda iv: -float(np.linalg.norm(iv[1])))[:2]
    v_a, v_b = edge_pairs[0][1], edge_pairs[1][1]
    ang_a = np.degrees(np.arctan2(v_a[1], v_a[0]))
    ang_b = np.degrees(np.arctan2(v_b[1], v_b[0]))
    diff = abs(((ang_a - ang_b + 90.0) % 180.0) - 90.0)
    if diff > max_edge_angle_diff_deg:
        reasons.append(f"edges_angle_diff {diff:.1f}°>{max_edge_angle_diff_deg}°")

    return (len(reasons) == 0), reasons


# ─────────────────────────────────────────────────────────────────────────────
# Ujjpár-szuppresszió
# ─────────────────────────────────────────────────────────────────────────────

def suppress_finger_pairs(fret_xs: list,
                          min_px: float = 8.0,
                          max_px: float = 22.0) -> tuple[list, list]:
    """Eltávolítja azokat a fret-jelölteket, amelyek 8–22 px-en belül párban vannak.

    Visszaad: (kept: list[float], removed_pairs: list[tuple])
    """
    if len(fret_xs) < 2:
        return list(fret_xs), []
    xs = np.array(sorted(fret_xs), dtype=np.float64)
    removed_idx = set()
    pairs = []
    for i in range(len(xs)):
        if i in removed_idx:
            continue
        for j in range(i + 1, len(xs)):
            if j in removed_idx:
                continue
            d = xs[j] - xs[i]
            if d > max_px:
                break
            if min_px <= d <= max_px:
                removed_idx.add(i)
                removed_idx.add(j)
                pairs.append((float(xs[i]), float(xs[j])))
                break
    kept = [float(x) for k, x in enumerate(xs) if k not in removed_idx]
    return kept, pairs


# ─────────────────────────────────────────────────────────────────────────────
# Nut oldal becslés anchor alapján
# ─────────────────────────────────────────────────────────────────────────────

def _choose_nut_side(anchor: Optional[dict],
                     H: Optional[np.ndarray],
                     img_shape: tuple) -> Optional[str]:
    """A kéz-anchor segítségével becsli, melyik oldalon van a nut."""
    if anchor is None or H is None:
        return None
    pc = anchor["palm_center_px"]
    pt = np.array([pc[0], pc[1], 1.0])
    proj = H @ pt
    if abs(proj[2]) < 1e-9:
        return None
    cx = float(proj[0] / proj[2])
    return "right" if cx < CANONICAL_W / 2.0 else "left"


# ─────────────────────────────────────────────────────────────────────────────
# run_v14_pipeline – a fő orchestrátor
# ─────────────────────────────────────────────────────────────────────────────

def run_v14_pipeline(img_entry: dict,
                     landmarker=None) -> dict:
    """Egy képre lefuttatja a V14 pipeline-t.

    Args:
        img_entry: dict {'path', 'class', ...} – általában manifest sor.
        landmarker: HandLandmarker vagy None (ilyenkor lazy singleton).

    Visszaad: dict minden közbülső artefaktummal + 'ok' flag.
    """
    if landmarker is None:
        landmarker = _get_landmarker()

    out = {
        "class": img_entry.get("class", "?"),
        "path": img_entry["path"],
        "fname": img_entry.get("fname", str(img_entry["path"]).split("/")[-1]),
        "ok": False,
        "invalid_reason": None,
    }

    # ── Kép betöltés ────────────────────────────────────────────────────────
    try:
        img = load_image_bgr(img_entry["path"])
    except FileNotFoundError as e:
        out["invalid_reason"] = "load_failed"
        return out
    out["img"] = img

    # ── 1. MediaPipe landmarks ───────────────────────────────────────────────
    landmarks = step9_detect_landmarks(img_entry["path"], landmarker)
    out["landmarks"] = landmarks

    # ── 2. Anchor + ujjmaszk ────────────────────────────────────────────────
    anchor = anchor_neck_angle(landmarks, img.shape)
    out["anchor"] = anchor
    finger_mask = build_finger_mask(img.shape, landmarks)
    out["finger_mask"] = finger_mask

    # ── 3. Canny + maszkolás ────────────────────────────────────────────────
    edges = step1_canny(img)
    edges_masked = edges.copy()
    if finger_mask.any():
        edges_masked[finger_mask > 0] = 0
    out["edges"] = edges
    out["edges_masked"] = edges_masked

    # ── 4. Hough ────────────────────────────────────────────────────────────
    lines = step2_hough(img, edges_masked)
    out["lines"] = lines
    if not lines:
        out["invalid_reason"] = "no_hough_lines"
        return out

    # ── 5. Nyakirány: plain elsőként, anchor csak ha szükséges ─────────────
    neck_plain = step3_neck_angle(lines)
    split_plain = step4_split_lines(lines, neck_plain["angle_deg"])
    if len(split_plain["long_lines"]) >= 3 or anchor is None:
        neck, split = neck_plain, split_plain
        neck["anchor_used"] = False
    else:
        neck_anc = step3_neck_angle_anchored(lines, anchor=anchor)
        split_anc = step4_split_lines(lines, neck_anc["angle_deg"])
        if len(split_anc["long_lines"]) > len(split_plain["long_lines"]):
            neck, split = neck_anc, split_anc
            neck["anchor_used"] = True
        else:
            neck, split = neck_plain, split_plain
            neck["anchor_used"] = False

    out["neck"] = neck
    out["split"] = split

    if not split["long_lines"]:
        out["invalid_reason"] = "no_long_lines"
        return out

    # ── 6. Outer edges ──────────────────────────────────────────────────────
    edge_info = step5_outer_edges(split["long_lines"], neck["angle_deg"])
    out["edge_info"] = edge_info
    if edge_info is None:
        out["invalid_reason"] = "no_outer_edges"
        return out

    # ── 7. Trapézoid ────────────────────────────────────────────────────────
    trap = step6_trapezoid(img, edge_info)
    out["trap"] = trap
    if trap is None:
        out["invalid_reason"] = "no_trapezoid"
        return out

    # ── 8. Trapézoid szanitás ───────────────────────────────────────────────
    ok, reasons = validate_trapezoid(trap["corners_px"], img.shape, landmarks)
    out["trap_ok"] = ok
    out["trap_reasons"] = reasons
    if not ok:
        out["invalid_reason"] = "trapezoid_sanity: " + ", ".join(reasons)
        return out

    # ── 9. Warp ─────────────────────────────────────────────────────────────
    H, H_inv, canon = step6_warp(img, trap["corners_px"])
    out["H"], out["H_inv"], out["canon"] = H, H_inv, canon

    # ── 10. Nut detektálás + anchor override ────────────────────────────────
    side_hint = _choose_nut_side(anchor, H, img.shape)
    out["nut_side_hint"] = side_hint
    nut = step6b_find_nut(canon)

    if side_hint is not None and (nut is None or nut.get("side") != side_hint):
        gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        col_response = np.abs(sx).sum(axis=0).astype(np.float32)
        w = canon.shape[1]
        sw = max(int(w * 0.30), 10)
        min_off = 5
        med = float(np.median(col_response))
        if side_hint == "left":
            region = col_response[min_off:sw]
            nx = int(np.argmax(region)) + min_off
            peak = float(region.max())
        else:
            region = col_response[w - sw:w - min_off]
            nx = (w - sw) + int(np.argmax(region))
            peak = float(region.max())
        ratio = peak / (med + 1e-6)
        if ratio >= 2.0:
            nut = {"side": side_hint, "nut_x": nx, "peak": peak,
                   "ratio": ratio, "col_response": col_response,
                   "from_anchor_override": True}

    out["nut"] = nut

    # ── 11. Nut-trim + re-warp ──────────────────────────────────────────────
    if nut is not None:
        corners_trim = step6c_trim_to_nut(trap["corners_px"], H_inv, nut)
        H2, H2_inv, canon2 = step6_warp(img, corners_trim)
        out["corners_trim"] = corners_trim
        out["H"], out["H_inv"], out["canon"] = H2, H2_inv, canon2

    # ── 12. Bundvonalak detektálása ─────────────────────────────────────────
    fret_xs_raw = step7_fret_lines_canonical(out["canon"])
    out["fret_xs_raw"] = fret_xs_raw

    # ── 13. Ujjpár szuppresszió ─────────────────────────────────────────────
    fret_xs_filt, removed_pairs = suppress_finger_pairs(fret_xs_raw)
    out["fret_xs_filt"] = fret_xs_filt
    out["removed_pairs"] = removed_pairs

    # ── 14. 17.817-es illesztés ─────────────────────────────────────────────
    try:
        nut_side = nut["side"] if nut else None
        fit = step8_fit_fret_rule(
            fret_xs_filt,
            nut_anchored=(nut_side is not None),
            nut_side=nut_side,
        )
        out["fit"] = fit
    except Exception as exc:
        out["fit"] = None
        out["invalid_reason"] = f"step8_failed: {exc}"
        return out

    # ── 15. Ujjhegy vetítés (ha van landmark és H) ──────────────────────────
    h_img, w_img = img.shape[:2]
    out["fingertips"] = step9_project_fingertips(
        landmarks, out["H"], w_img, h_img, fit=fit
    )

    out["ok"] = True
    return out
