"""
src/geometry.py

Összes OpenCV-alapú fretboard geometria: Canny → Hough → nyakirány →
vonalszétválasztás → külső nyakélek → trapézoid → warp → nut detektálás →
bundvonalak → 17.817-es szabály illesztés.

Forrás: 03c_pipeline_fixes_design.ipynb (V14), cellák 8–16.
Megjegyzés: _score_inlay_fit a v13 notebookból van visszaépítve (03c-ből kimaradt).
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from src.config import CFG
from src.constants import (
    CANONICAL_H, CANONICAL_W,
    FRET_POS_NORM, FRET_POS_FULL,
    N_FRETS, INLAY_NORM_DICT,
)

# Helyi konstansok a fret illesztőhöz
_TARGET_RATIO = 2.0 ** (1.0 / 12.0)   # ≈ 1.05946
_MIN_RUN_SPACINGS = 2                   # min 2 szomszédos arány = 3 pont


# ─────────────────────────────────────────────────────────────────────────────
# Segédfüggvények
# ─────────────────────────────────────────────────────────────────────────────

def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1].copy()


def load_image_bgr(path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Kép nem olvasható: {path}")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Canny éldetektálás
# ─────────────────────────────────────────────────────────────────────────────

def step1_canny(img_bgr: np.ndarray,
                low: int = CFG["canny_low"],
                high: int = CFG["canny_high"],
                blur_ksize: int = CFG["canny_blur_ksize"]) -> np.ndarray:
    """Canny éldetektálás az egész képen. Visszaad: bináris élkép (uint8)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur_ksize >= 3:
        blur_ksize = blur_ksize if blur_ksize % 2 else blur_ksize + 1
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return cv2.Canny(gray, low, high)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – HoughLinesP
# ─────────────────────────────────────────────────────────────────────────────

def step2_hough(img_bgr: np.ndarray,
                edges: np.ndarray,
                threshold: int = CFG["hough_threshold"],
                min_len_frac: float = CFG["hough_min_len_frac"],
                max_gap: int = CFG["hough_max_gap"]) -> list[tuple]:
    """HoughLinesP a teljes képen. Visszaad: [(x1,y1,x2,y2), ...] lista."""
    h, w = img_bgr.shape[:2]
    min_len = int(min(h, w) * min_len_frac)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=threshold, minLineLength=min_len, maxLineGap=max_gap,
    )
    if lines is None:
        return []
    return [tuple(map(int, ln[0])) for ln in lines]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Nyakirány meghatározása
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_angle(angle_deg: float) -> float:
    """[-90, +90) tartományra normalizál."""
    a = float(angle_deg % 180.0)
    if a >= 90.0:
        a -= 180.0
    return a


def _line_stats(line: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = line
    angle = _normalize_angle(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    length = float(np.hypot(x2 - x1, y2 - y1))
    return angle, length


def step3_neck_angle(lines: list) -> dict:
    """Hosszal súlyozott szöghisztogram → domináns nyakirány."""
    if not lines:
        return {"angle_deg": 0.0, "hist": np.array([]), "bin_edges": np.array([]),
                "angles": np.array([]), "lengths": np.array([])}
    angles = np.array([_line_stats(l)[0] for l in lines])
    lengths = np.array([_line_stats(l)[1] for l in lines])
    hist, bin_edges = np.histogram(
        angles, bins=36, range=(-90.0, 90.0), weights=lengths
    )
    peak = int(np.argmax(hist))
    lo = max(0, peak - 1)
    hi = min(len(hist) - 1, peak + 1)
    centers = (bin_edges[lo:hi + 2:1][:-1] + bin_edges[lo:hi + 2:1][1:]) / 2.0
    w = hist[lo:hi + 1]
    angle_deg = float(np.dot(w, centers) / w.sum()) if w.sum() > 0 else float(
        (bin_edges[peak] + bin_edges[peak + 1]) / 2
    )
    return {"angle_deg": angle_deg, "hist": hist, "bin_edges": bin_edges,
            "angles": angles, "lengths": lengths}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Hosszanti vs. bund-irányú vonalak szétválasztása
# ─────────────────────────────────────────────────────────────────────────────

def step4_split_lines(lines: list, neck_angle_deg: float,
                      angle_tol: int = 20) -> dict:
    """Hosszanti (neck-párhuzamos) és bund-irányú (neck-merőleges) vonalak."""
    neck_angle = _normalize_angle(neck_angle_deg)
    long_lines, fret_lines, other = [], [], []
    for line in lines:
        ang, _ = _line_stats(line)
        d_long = min(abs(ang - neck_angle), 180.0 - abs(ang - neck_angle))
        d_fret = abs(d_long - 90.0)
        if d_long <= angle_tol:
            long_lines.append(line)
        elif d_fret <= angle_tol:
            fret_lines.append(line)
        else:
            other.append(line)
    return {"long_lines": long_lines, "fret_lines": fret_lines, "other": other}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Külső nyakélek detektálása (outlier-alapú, v9)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_cluster_edge(cluster_items: list,
                      perp_dir: np.ndarray,
                      neck_dir: np.ndarray) -> dict:
    """Hossz-súlyozott centroid az él-klaszterből."""
    all_pts, weights = [], []
    for item in cluster_items:
        x1, y1, x2, y2 = item["line"]
        seg_len = float(np.hypot(x2 - x1, y2 - y1)) + 1e-6
        all_pts.extend([np.array([x1, y1], dtype=np.float64),
                        np.array([x2, y2], dtype=np.float64)])
        weights.extend([seg_len, seg_len])
    all_pts = np.array(all_pts)
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    centroid = (all_pts * w[:, None]).sum(axis=0)
    proj = float(np.dot(centroid, perp_dir))
    along = float(np.dot(centroid, neck_dir))
    longest = max(cluster_items,
                  key=lambda it: np.hypot(it["line"][2] - it["line"][0],
                                          it["line"][3] - it["line"][1]))
    return {"midpoint": centroid, "proj": proj, "along": along,
            "line": longest["line"]}


def _find_neck_edge_outliers(projs_sorted: list,
                             outlier_ratio: float = 2.5) -> tuple:
    """Izolált nyakél-vonalak a vetületek szélein."""
    if len(projs_sorted) < 3:
        return None, None
    proj_vals = np.array([p["proj"] for p in projs_sorted])
    gaps = np.diff(proj_vals)
    if len(gaps) < 2:
        return None, None
    left_outlier = right_outlier = None
    rest_median_right = float(np.median(gaps[1:]))
    if rest_median_right > 1e-3 and gaps[0] > outlier_ratio * rest_median_right:
        left_outlier = projs_sorted[0]
    rest_median_left = float(np.median(gaps[:-1]))
    if rest_median_left > 1e-3 and gaps[-1] > outlier_ratio * rest_median_left:
        right_outlier = projs_sorted[-1]
    return left_outlier, right_outlier


def step5_outer_edges(long_lines: list,
                      neck_angle_deg: float,
                      angle_tol: int = CFG["step5_angle_tol"],
                      outlier_ratio: float = CFG["step5_outlier_ratio"],
                      expansion_margin_frac: float = CFG["step5_expansion_margin_frac"]
                      ) -> Optional[dict]:
    """Két külső nyakél – outlier-alapú detektálás (v9)."""
    if not long_lines:
        return None

    neck_rad = np.deg2rad(neck_angle_deg)
    neck_dir = np.array([np.cos(neck_rad), np.sin(neck_rad)], dtype=np.float64)
    perp_dir = np.array([-neck_dir[1], neck_dir[0]], dtype=np.float64)

    all_pts = ([(x1, y1) for x1, y1, x2, y2 in long_lines] +
               [(x2, y2) for x1, y1, x2, y2 in long_lines])
    img_span = float(np.max([np.hypot(p[0], p[1]) for p in all_pts]))
    cluster_gap = max(img_span * 0.025, 12.0)

    projs = []
    for line in long_lines:
        x1, y1, x2, y2 = line
        mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        projs.append({"line": line, "midpoint": mid.copy(),
                      "proj": float(np.dot(mid, perp_dir)),
                      "along": float(np.dot(mid, neck_dir))})

    projs_sorted = sorted(projs, key=lambda p: p["proj"])
    left_outlier, right_outlier = _find_neck_edge_outliers(
        projs_sorted, outlier_ratio=outlier_ratio
    )
    left_is_outlier = left_outlier is not None
    right_is_outlier = right_outlier is not None

    print(f"  [outer_edges_v9] vonalak: {len(projs)} | "
          f"bal outlier: {'igen' if left_is_outlier else 'nem'} | "
          f"jobb outlier: {'igen' if right_is_outlier else 'nem'}")

    inner = [p for p in projs_sorted
             if p is not left_outlier and p is not right_outlier]
    if not inner:
        inner = projs_sorted

    left_cl = [inner[0]]
    for item in inner[1:]:
        if abs(item["proj"] - left_cl[-1]["proj"]) <= cluster_gap:
            left_cl.append(item)
        else:
            break

    right_cl = [inner[-1]]
    for item in reversed(inner[:-1]):
        if abs(item["proj"] - right_cl[-1]["proj"]) <= cluster_gap:
            right_cl.append(item)
        else:
            break

    inner_lines = []
    seen = set()
    for itm in (left_cl + right_cl):
        key = tuple(itm["line"])
        if key not in seen:
            inner_lines.append(itm)
            seen.add(key)

    left_inner_rep = _fit_cluster_edge(left_cl, perp_dir, neck_dir)
    right_inner_rep = _fit_cluster_edge(right_cl, perp_dir, neck_dir)

    inner_projs = np.array([p["proj"] for p in inner])
    cluster_width = float(inner_projs[-1] - inner_projs[0]) if len(inner_projs) > 1 else 0.0
    margin = cluster_width * expansion_margin_frac
    expansion_margin = 0.0

    if left_is_outlier:
        left_rep = dict(left_outlier)
    else:
        left_rep = left_inner_rep
        left_rep["midpoint"] = left_rep["midpoint"] - margin * perp_dir
        left_rep["proj"] -= margin
        expansion_margin = margin
        print(f"  [outer_edges_v9] Bal oldal bővítve: -{margin:.1f}px")

    if right_is_outlier:
        right_rep = dict(right_outlier)
    else:
        right_rep = right_inner_rep
        right_rep["midpoint"] = right_rep["midpoint"] + margin * perp_dir
        right_rep["proj"] += margin
        expansion_margin = margin
        print(f"  [outer_edges_v9] Jobb oldal bővítve: +{margin:.1f}px")

    final_sep = abs(right_rep["proj"] - left_rep["proj"])
    print(f"  [outer_edges_v9] szétválasztás: {final_sep:.1f}px "
          f"(klaszter: {cluster_width:.1f}px | bővítés: {expansion_margin:.1f}px)")

    return {
        "neck_dir": neck_dir,
        "perp_dir": perp_dir,
        "selected_edges": [left_rep, right_rep],
        "all_projections": projs,
        "inner_lines": inner_lines,
        "cluster_gap": cluster_gap,
        "left_is_outlier": left_is_outlier,
        "right_is_outlier": right_is_outlier,
        "expansion_margin": expansion_margin,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Trapézoid + perspektíva warp
# ─────────────────────────────────────────────────────────────────────────────

def _pt_on_line(midpoint: np.ndarray, neck_dir: np.ndarray,
                target_along: float) -> np.ndarray:
    """Egy vonalon lévő pont, amelynek along-vetülete = target_along."""
    base = float(np.dot(midpoint, neck_dir))
    return midpoint + (target_along - base) * neck_dir


def step6_clamp_trapezoid_extent(edge_info: dict,
                                 anchor: Optional[dict],
                                 margin_px: int = 30) -> dict:
    """Korlátozza a trapézoid along-kiterjedését a csukló pozíciójához.

    A csukló a Nut közelében van; az alkar a csukló túloldalán nyúlik el.
    Meghatározza, hogy a csukló az a_min vagy az a_max oldalhoz közelebb
    van-e, majd azt az oldalt korlátozza: a trapézoid nem nyúlhat az alkar
    felé margin_px-nél többel a csukló mögé.
    """
    if anchor is None or edge_info is None:
        return edge_info
    neck_dir = edge_info["neck_dir"]
    wrist_px = anchor.get("wrist_px")
    if wrist_px is None:
        return edge_info
    wrist_along = float(np.dot(np.array(wrist_px, dtype=np.float64), neck_dir))
    edge_info = dict(edge_info)
    edge_info["wrist_along"] = wrist_along
    edge_info["clamp_margin_px"] = margin_px
    return edge_info


def step6_trapezoid(img_bgr: np.ndarray, edge_info: dict) -> Optional[dict]:
    """A két kiválasztott él alapján felépíti a fretboard trapézot (v9)."""
    if edge_info is None:
        return None
    edges = edge_info["selected_edges"]
    neck_dir = edge_info["neck_dir"]
    perp_dir = edge_info["perp_dir"]
    if len(edges) < 2:
        return None

    left, right = sorted(edges, key=lambda e: e["proj"])

    inner = edge_info.get("inner_lines") or edge_info.get("all_projections")
    all_endpts = []
    for item in inner:
        x1, y1, x2, y2 = item["line"]
        all_endpts.append(np.array([x1, y1], dtype=np.float64))
        all_endpts.append(np.array([x2, y2], dtype=np.float64))
    alongs = [float(np.dot(p, neck_dir)) for p in all_endpts]
    a_min = min(alongs)
    a_max = max(alongs)

    # Clamp: a csukló pozíciója meghatározza a Nut-oldali határt.
    # A csukló (nut-oldal) felé nem nyúlhat a trapézoid margin_px-nél tovább.
    wrist_along = edge_info.get("wrist_along")
    if wrist_along is not None:
        margin = edge_info.get("clamp_margin_px", 30)
        mid = (a_min + a_max) / 2.0
        if wrist_along < mid:
            # Csukló az a_min oldalon → a_min nem mehet wrist_along - margin alá
            a_min = max(a_min, wrist_along - margin)
        else:
            # Csukló az a_max oldalon → a_max nem mehet wrist_along + margin fölé
            a_max = min(a_max, wrist_along + margin)

    span = a_max - a_min
    a_min -= span * 0.02
    a_max += span * 0.02

    l_start = _pt_on_line(left["midpoint"], neck_dir, a_min)
    l_end = _pt_on_line(left["midpoint"], neck_dir, a_max)
    r_start = _pt_on_line(right["midpoint"], neck_dir, a_min)
    r_end = _pt_on_line(right["midpoint"], neck_dir, a_max)

    w_start = float(np.linalg.norm(r_start - l_start))
    w_end = float(np.linalg.norm(r_end - l_end))

    hysteresis = 0.05 * min(w_start, w_end)
    if (w_end - w_start) >= -hysteresis:
        tl, tr, br, bl = l_start, l_end, r_end, r_start
    else:
        tl, tr, br, bl = l_end, l_start, r_start, r_end

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    area = 0.5 * abs(
        np.dot(corners[:, 0], np.roll(corners[:, 1], -1)) -
        np.dot(corners[:, 1], np.roll(corners[:, 0], -1))
    )
    print(f"  [trapezoid_v9] span={span:.1f}px | "
          f"w_start={w_start:.1f}px | w_end={w_end:.1f}px | area={area:.0f}px²")
    return {"corners_px": corners, "area_px2": float(area),
            "w_start": w_start, "w_end": w_end}


def step6_warp(img_bgr: np.ndarray,
               corners_px: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perspektíva-warp a trapézból a kanonikus térbe. Visszaad: (H, H_inv, canon_bgr)."""
    W, H = CFG["canonical_w"], CFG["canonical_h"]
    dst = np.array([
        [0,     0    ],
        [W - 1, 0    ],
        [W - 1, H - 1],
        [0,     H - 1],
    ], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(corners_px.astype(np.float32), dst)
    mat_inv = cv2.getPerspectiveTransform(dst, corners_px.astype(np.float32))
    canon = cv2.warpPerspective(img_bgr, mat, (W, H))
    return mat, mat_inv, canon


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6b/6c – Nut detektálás
# ─────────────────────────────────────────────────────────────────────────────

def _measure_peak_fwhm(profile: np.ndarray, peak_x: int) -> float:
    """Full Width at Half Maximum a profil `peak_x` pozíciójánál.

    A Nut fizikailag 1.5–2× vastagabb mint egy bund → FWHM > ~5px.
    Bundcsúcsoknál FWHM tipikusan 2–4px.

    Visszaad: FWHM pixelben (float). Ismeretlen/szélső esetén: 0.0.
    """
    n = len(profile)
    if peak_x <= 0 or peak_x >= n - 1:
        return 0.0
    half = float(profile[peak_x]) * 0.5
    # Bal oldal: hol esik a profil a félérték alá
    left = peak_x
    while left > 0 and profile[left] >= half:
        left -= 1
    # Jobb oldal: hol esik a profil a félérték alá
    right = peak_x
    while right < n - 1 and profile[right] >= half:
        right += 1
    return float(right - left)


def step6b_find_nut(canon_bgr: np.ndarray,
                    search_frac: float = 0.30,
                    threshold_factor: float = 2.5,
                    min_offset: int = 5,
                    side_hint: Optional[str] = None,
                    hand_boundary_canon_x: Optional[float] = None) -> Optional[dict]:
    """Megkeresi a nutot (0. bund) a kanonikus képen Sobel-x alapján.

    Ha side_hint ('left'/'right') adott, csak azt az oldalt vizsgálja
    kiterjesztett (40%) keresési sávval és alacsonyabb (2.0×) küszöbbel.

    Ha hand_boundary_canon_x megadott (CFG['hand_boundary_enabled']),
    a keresési sáv felső határát `hand_boundary_canon_x - nut_hand_margin_px`-re
    korlátozza, így a kézen belül nem keres Nut-jelöltet.

    Ha CFG['nut_width_filter_enabled'], top-N jelöltet vizsgál FWHM alapján
    (argmax helyett), és csak a minimális szélességet meghaladó csúcsot fogadja el.
    """
    from scipy.signal import find_peaks as _find_peaks

    gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    col_response = np.abs(sx).sum(axis=0).astype(np.float32)

    w = canon_bgr.shape[1]
    median_response = float(np.median(col_response))
    width_filter = bool(CFG.get("nut_width_filter_enabled", True))
    min_fwhm = float(CFG.get("nut_min_width_px", 5.0))
    n_cand = int(CFG.get("nut_n_candidates", 5))
    margin = int(CFG.get("nut_hand_margin_px", 10))
    hand_bnd_enabled = bool(CFG.get("hand_boundary_enabled", True))

    def _clamp_sw(sw: int, is_left: bool) -> int:
        """Korlátozza a keresési sáv végét a kézél előtt."""
        if not (hand_bnd_enabled and hand_boundary_canon_x is not None):
            return sw
        if is_left:
            limit = int(hand_boundary_canon_x) - margin
            return min(sw, max(min_offset + 1, limit))
        else:
            limit = w - int(hand_boundary_canon_x) - margin
            return min(sw, max(min_offset + 1, limit))

    def _best_candidate(region: np.ndarray, offset: int) -> tuple[int, float, float]:
        """Top-N csúcs közül FWHM-szűréssel a legjobb jelölt.

        Visszaad: (local_argmax, peak_val, fwhm_px)
        """
        if not width_filter or len(region) < 3:
            idx = int(np.argmax(region))
            fwhm = _measure_peak_fwhm(col_response, idx + offset)
            return idx, float(region[idx]), fwhm

        # find_peaks a régión – top-N a magasság szerint
        idxs, props = _find_peaks(region, height=0.0)
        if len(idxs) == 0:
            idx = int(np.argmax(region))
            fwhm = _measure_peak_fwhm(col_response, idx + offset)
            return idx, float(region[idx]), fwhm

        heights = region[idxs]
        top_n = idxs[np.argsort(heights)[::-1][:n_cand]]
        # Szélességi szűrés: az első (legmagasabb) n_cand csúcs közül
        # az első, amelynek FWHM >= min_fwhm
        for ci in top_n:
            fwhm = _measure_peak_fwhm(col_response, int(ci) + offset)
            if fwhm >= min_fwhm:
                return int(ci), float(region[ci]), fwhm
        # Ha egyik sem elég széles, visszaesünk a legmagasabbra
        ci = top_n[0]
        fwhm = _measure_peak_fwhm(col_response, int(ci) + offset)
        return int(ci), float(region[ci]), fwhm

    if side_hint is not None:
        # Egyoldalas keresés: kiterjesztett régió, lazább küszöb
        sw_raw = max(int(w * 0.40), 10)
        thr = 2.0 * (median_response + 1e-6)
        if side_hint == "left":
            sw = _clamp_sw(sw_raw, is_left=True)
            region = col_response[min_offset:sw]
            if len(region) == 0:
                return None
            local_idx, peak, fwhm = _best_candidate(region, min_offset)
            nut_x = local_idx + min_offset
        else:
            sw = _clamp_sw(sw_raw, is_left=False)
            region = col_response[w - sw:w - min_offset]
            if len(region) == 0:
                return None
            local_idx, peak, fwhm = _best_candidate(region, w - sw)
            nut_x = (w - sw) + local_idx

        ratio = peak / (median_response + 1e-6)
        print(f"  [nut_detect_v12] side_hint={side_hint} | "
              f"median={median_response:.0f} | peak={peak:.0f} | "
              f"ratio={ratio:.2f} | fwhm={fwhm:.1f}px")
        if peak < thr:
            print(f"  [nut_detect_v12] nincs egyértelmű nut (ratio={ratio:.2f} < 2.0)")
            return None
        print(f"  [nut_detect_v12] nut találat: {side_hint} @ x={nut_x}px (fwhm={fwhm:.1f}px)")
        return {"side": side_hint, "nut_x": nut_x, "peak": peak, "ratio": ratio,
                "width_px": fwhm, "col_response": col_response}

    # Kétoldalas fallback (side_hint=None esetén)
    sw_raw = max(int(w * search_frac), 10)
    sw_l = _clamp_sw(sw_raw, is_left=True)
    sw_r = _clamp_sw(sw_raw, is_left=False)
    left_region = col_response[min_offset:sw_l]
    right_region = col_response[w - sw_r:w - min_offset]

    if len(left_region) == 0 or len(right_region) == 0:
        return None

    l_idx, left_peak_val, l_fwhm = _best_candidate(left_region, min_offset)
    r_idx, right_peak_val, r_fwhm = _best_candidate(right_region, w - sw_r)
    threshold = threshold_factor * (median_response + 1e-6)

    if left_peak_val >= right_peak_val:
        side, peak, nut_x, fwhm = "left", left_peak_val, l_idx + min_offset, l_fwhm
    else:
        side, peak, nut_x, fwhm = "right", right_peak_val, (w - sw_r) + r_idx, r_fwhm

    ratio = peak / (median_response + 1e-6)
    print(f"  [nut_detect_v12] median={median_response:.0f} | "
          f"left_peak={left_peak_val:.0f}(w={l_fwhm:.1f}) | "
          f"right_peak={right_peak_val:.0f}(w={r_fwhm:.1f})")

    if peak < threshold:
        print(f"  [nut_detect_v12] nincs egyértelmű nut (peak/median={ratio:.2f} < {threshold_factor})")
        return None

    print(f"  [nut_detect_v12] nut találat: {side} @ x={nut_x}px "
          f"(ratio={ratio:.2f}, fwhm={fwhm:.1f}px)")
    return {"side": side, "nut_x": nut_x, "peak": peak, "ratio": ratio,
            "width_px": fwhm, "col_response": col_response}


def step6c_trim_to_nut(corners_px: np.ndarray,
                       H_inv: np.ndarray,
                       nut_info: dict) -> np.ndarray:
    """Trapezoid sarokpontok áthelyezése a nut canonical x-pozíciójához."""
    nut_x = nut_info["nut_x"]
    side = nut_info["side"]
    H = CFG["canonical_h"]
    canon_pts = np.array([[nut_x, 0],
                          [nut_x, H - 1]], dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(canon_pts, H_inv).reshape(-1, 2)
    new_corners = corners_px.copy().astype(np.float32)
    if side == "left":
        new_corners[0] = img_pts[0]
        new_corners[3] = img_pts[1]
    else:
        new_corners[1] = img_pts[0]
        new_corners[2] = img_pts[1]
    return new_corners


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – Bundvonalak a kanonikus képen
# ─────────────────────────────────────────────────────────────────────────────

def _column_variance_frets(canon_bgr: np.ndarray,
                           min_height: float = 0.18,
                           min_distance: int = 10,
                           smooth_kernel: int = 5) -> list[float]:
    """Fallback bunddetektálás oszlopvariancia alapján."""
    gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    col_std = np.std(gray, axis=0)
    max_val = col_std.max()
    if max_val < 1e-3:
        return []
    col_norm = col_std / max_val
    k = smooth_kernel if smooth_kernel % 2 else smooth_kernel + 1
    smoothed = np.convolve(col_norm, np.ones(k) / k, mode="same")

    peaks = []
    try:
        from scipy.signal import find_peaks as _fp
        idxs, _ = _fp(smoothed, height=min_height, distance=min_distance)
        peaks = [float(i) for i in idxs]
    except ImportError:
        for i in range(1, len(smoothed) - 1):
            if (smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
                    and smoothed[i] > min_height):
                peaks.append(float(i))
        filtered = []
        for p in peaks:
            if not filtered or p - filtered[-1] >= min_distance:
                filtered.append(p)
        peaks = filtered

    print(f"  [col_var_fallback] csúcsok: {len(peaks)}")
    return peaks


def _filter_wide_clusters(clusters: list,
                           max_width_frac: float,
                           max_fret_width_px: float) -> tuple[list, list]:
    """Szélességalapú klaszterszűrő (v12) – ujjak kizárása."""
    if len(clusters) < 2:
        return clusters, []
    means = sorted([float(np.mean(c)) for c in clusters])
    spacings = np.diff(means)
    median_spacing = float(np.median(spacings)) if len(spacings) > 0 else float("inf")
    adaptive_limit = max_width_frac * median_spacing if median_spacing > 1e-3 else float("inf")
    width_limit = min(adaptive_limit, max_fret_width_px)

    kept, rejected_widths = [], []
    for c in clusters:
        w = float(max(c) - min(c)) if len(c) > 1 else 0.0
        if w <= width_limit:
            kept.append(c)
        else:
            rejected_widths.append(w)
            print(f"  [step7_v12] Klaszter kizárva (szélesség {w:.1f}px > limit {width_limit:.1f}px)")

    if not kept:
        print("  [step7_v12] ⚠ Összes klaszter kizárásra kerülne → szűrő kikapcsolva")
        return clusters, []
    return kept, rejected_widths


def step7_fret_lines_canonical(canon_bgr: np.ndarray,
                               canny_low: int = 15,
                               canny_high: int = 75,
                               threshold: int = 18,
                               min_len_frac: float = 0.25,
                               max_gap: int = 8,
                               angle_tol_from_vert: float = 45.0,
                               cluster_gap: float = 15.0,
                               var_fallback: bool = True,
                               var_min_height: float = 0.18,
                               var_min_distance: int = 10,
                               max_width_frac: float = 0.4,
                               max_fret_width_px: float = 18.0) -> list[float]:
    """Bundvonalak detektálása a kanonikus 600×80 px képen (v5/v12)."""
    H = CFG["canonical_h"]
    gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=threshold,
        minLineLength=max(int(H * min_len_frac), 6),
        maxLineGap=max_gap,
    )

    xs_all, xs_ok = [], []
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = map(int, ln[0])
            angle_v = abs(90.0 - abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            xs_all.append((x1 + x2) / 2.0)
            if angle_v < angle_tol_from_vert:
                xs_ok.append((x1 + x2) / 2.0)

    print(f"  [step7] Hough: {len(xs_all)} nyers vonal → szűrve: {len(xs_ok)}")

    if xs_ok:
        xs_ok.sort()
        clusters = [[xs_ok[0]]]
        for x in xs_ok[1:]:
            if abs(x - clusters[-1][-1]) < cluster_gap:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        clusters, rejected = _filter_wide_clusters(clusters, max_width_frac, max_fret_width_px)
        result = [float(np.mean(c)) for c in clusters]
        print(f"  [step7] HoughLinesP → {len(result)} klaszter ({len(rejected)} széles kizárva)")
        return result

    if var_fallback:
        print("  [step7] Hough 0 vonalat adott → oszlopvariancia fallback")
        return _column_variance_frets(
            canon_bgr, min_height=var_min_height, min_distance=var_min_distance
        )
    return []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 – 17.817-es bund-illesztés (v11/v13)
# ─────────────────────────────────────────────────────────────────────────────

def _ratio_runs(xs: np.ndarray,
                target: float = _TARGET_RATIO,
                tol: float = 0.06,
                direction: str = "forward") -> list[tuple[int, int]]:
    """Konzisztens spacing-ratio futamok keresése."""
    if len(xs) < 3:
        return []
    spacings = np.diff(xs)
    good = []
    for i in range(len(spacings) - 1):
        s_curr, s_next = spacings[i], spacings[i + 1]
        if s_next < 1e-3 or s_curr < 1e-3:
            good.append(False)
            continue
        r = (s_curr / s_next) if direction == "forward" else (s_next / s_curr)
        good.append(abs(r - target) < tol)
    runs, i = [], 0
    while i < len(good):
        if good[i]:
            j = i
            while j < len(good) and good[j]:
                j += 1
            if (j - i + 1) >= _MIN_RUN_SPACINGS:
                runs.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return runs


def _evaluate_fit(xs: np.ndarray, offset: float, scale: float,
                  tol_px: float) -> tuple[dict, list, int]:
    """Egy (offset, scale) pár inlierei + n_visible."""
    match, residuals = {}, []
    W = CFG["canonical_w"]
    for n in range(N_FRETS + 1):
        x_exp = offset + scale * FRET_POS_NORM[n]
        if x_exp < -W * 0.5 or x_exp > W * 1.5:
            continue
        dists = np.abs(xs - x_exp)
        min_d = dists.min()
        if min_d < tol_px:
            match[n] = float(xs[dists.argmin()])
            residuals.append(float(min_d))
    n_visible = sum(
        1 for n in range(N_FRETS + 1)
        if 0 <= offset + scale * FRET_POS_NORM[n] <= W
    )
    return match, residuals, n_visible


def _fit_from_run(xs: np.ndarray, run_start: int, run_end: int,
                  tol_px: float, scale_min: float, scale_max: float) -> Optional[dict]:
    """Konzisztens futamból (scale, offset, fret_start) – coverage × nut_prior."""
    run_xs = xs[run_start:run_end + 1]
    k = len(run_xs) - 1
    best, best_score, best_res = None, 0.0, float("inf")
    for n_start in range(N_FRETS - k + 1):
        n_end = n_start + k
        dp = FRET_POS_NORM[n_end] - FRET_POS_NORM[n_start]
        if abs(dp) < 1e-9:
            continue
        scale = (run_xs[-1] - run_xs[0]) / dp
        if not (scale_min <= scale <= scale_max):
            continue
        offset = run_xs[0] - scale * FRET_POS_NORM[n_start]
        match, residuals, n_visible = _evaluate_fit(xs, offset, scale, tol_px)
        n_in = len(match)
        avg_res = float(np.mean(residuals)) if residuals else float("inf")
        coverage = n_in / max(n_visible, 1)
        nut_prior = 1.0 - 0.7 * n_start / (N_FRETS + 1)
        score = coverage * nut_prior
        if score > best_score or (abs(score - best_score) < 0.01 and avg_res < best_res):
            best_score, best_res = score, avg_res
            best = {"matched_frets": match, "offset": offset, "scale": scale,
                    "avg_residual_px": avg_res, "fret_start": n_start,
                    "run_len": k + 1, "coverage_ratio": coverage,
                    "n_visible": n_visible, "nut_prior": nut_prior, "score": score}
    return best


def _fit_constrained_ransac(xs: np.ndarray, tol_px: float,
                            scale_min: float, scale_max: float,
                            max_anchors: int = 6) -> Optional[dict]:
    """Fallback: 2-param RANSAC scale tartomány-szűréssel."""
    anchor_xs = xs[:max_anchors]
    best, best_score, best_res = None, 0.0, float("inf")
    for i, xa in enumerate(anchor_xs):
        for j, xb in enumerate(anchor_xs):
            if j <= i:
                continue
            for na in range(N_FRETS + 1):
                for nb_ in range(na + 1, N_FRETS + 1):
                    dp = FRET_POS_NORM[nb_] - FRET_POS_NORM[na]
                    if abs(dp) < 1e-9:
                        continue
                    scale = (xb - xa) / dp
                    if not (scale_min <= scale <= scale_max):
                        continue
                    offset = xa - scale * FRET_POS_NORM[na]
                    match, residuals, n_visible = _evaluate_fit(xs, offset, scale, tol_px)
                    n_in = len(match)
                    avg_res = float(np.mean(residuals)) if residuals else float("inf")
                    coverage = n_in / max(n_visible, 1)
                    nut_prior = 1.0 - 0.7 * na / (N_FRETS + 1)
                    score = coverage * nut_prior
                    if score > best_score or (abs(score - best_score) < 0.01 and avg_res < best_res):
                        best_score, best_res = score, avg_res
                        best = {"matched_frets": match, "offset": offset, "scale": scale,
                                "avg_residual_px": avg_res, "fret_start": None,
                                "run_len": 0, "coverage_ratio": coverage,
                                "n_visible": n_visible, "nut_prior": nut_prior, "score": score}
    return best


def _score_inlay_fit(xs_inlays: list, offset: float, scale: float,
                     tol_px: float = 12.0) -> tuple[float, dict]:
    """Inlay egyezések számlálása (v13, visszaépítve v13 notebookból)."""
    if not xs_inlays or scale < 1e-3:
        return 0.0, {}
    W = CFG["canonical_w"]
    matched = {}
    for n, inorm in INLAY_NORM_DICT.items():
        x_exp = offset + scale * inorm
        if x_exp < 0 or x_exp > W:
            continue
        dists = [abs(xi - x_exp) for xi in xs_inlays]
        min_d = min(dists)
        if min_d < tol_px:
            matched[n] = xs_inlays[dists.index(min_d)]
    n_visible = sum(
        1 for inorm in INLAY_NORM_DICT.values()
        if 0 <= offset + scale * inorm <= W
    )
    score = float(len(matched)) / max(n_visible, 1)
    return score, matched


def _fit_with_inlay_anchor(xs_frets: np.ndarray,
                           xs_inlays: list,
                           nut_anchored: bool,
                           scale_min: float,
                           scale_max: float,
                           tol_px: float) -> Optional[dict]:
    """Inlay-kandid.-alapú illesztés (v13)."""
    total_det = len(xs_frets)
    best, best_score, best_res = None, 0.0, float("inf")
    inlay_items = list(INLAY_NORM_DICT.items())

    def _eval_candidate(offset_c, scale_c):
        nonlocal best, best_score, best_res
        if not (scale_min <= scale_c <= scale_max):
            return
        match, residuals, n_visible = _evaluate_fit(xs_frets, offset_c, scale_c, tol_px)
        n_in = len(match)
        if n_in < 1:
            return
        avg_res = float(np.mean(residuals)) if residuals else float("inf")
        covered = n_in / max(n_visible, 1)
        explained = n_in / max(total_det, 1)
        i_score, i_matched = _score_inlay_fit(xs_inlays, offset_c, scale_c, tol_px)
        score = covered * explained + 0.3 * i_score
        if score > best_score or (abs(score - best_score) < 0.005 and avg_res < best_res):
            best_score, best_res = score, avg_res
            best = {"matched_frets": match, "offset": offset_c, "scale": scale_c,
                    "avg_residual_px": avg_res, "fret_start": 0,
                    "run_len": 0, "coverage_ratio": covered,
                    "n_visible": n_visible, "nut_prior": 1.0,
                    "score": score, "explained": explained,
                    "inlay_score": i_score, "matched_inlays": i_matched}

    if nut_anchored:
        for xa in xs_inlays:
            if xa < 1.0:
                continue
            for n, inorm in inlay_items:
                if inorm < 1e-9:
                    continue
                _eval_candidate(0.0, xa / inorm)
    else:
        inlay_list = list(xs_inlays)
        for i_a, xa in enumerate(inlay_list):
            for xb in inlay_list[i_a + 1:]:
                for na, inorm_a in inlay_items:
                    for nb, inorm_b in inlay_items:
                        if nb <= na:
                            continue
                        dinorm = inorm_b - inorm_a
                        if abs(dinorm) < 1e-9:
                            continue
                        _eval_candidate(xa - (xb - xa) / dinorm * inorm_a,
                                        (xb - xa) / dinorm)
    return best


def _fit_with_nut_anchor(xs: np.ndarray, tol_px: float,
                         scale_min: float, scale_max: float) -> Optional[dict]:
    """v11.1: offset=0, fret_start=0 rögzített; csak scale-t keresi."""
    total_det = len(xs)
    best, best_score, best_res = None, 0.0, float("inf")
    for xi in xs:
        if xi < 1.0:
            continue
        for n in range(1, N_FRETS + 1):
            denom = FRET_POS_NORM[n]
            if denom < 1e-9:
                continue
            scale = xi / denom
            if not (scale_min <= scale <= scale_max):
                continue
            match, residuals, n_visible = _evaluate_fit(xs, 0.0, scale, tol_px)
            n_in = len(match)
            if n_in < 1:
                continue
            avg_res = float(np.mean(residuals)) if residuals else float("inf")
            covered = n_in / max(n_visible, 1)
            explained = n_in / max(total_det, 1)
            score = covered * explained
            if score > best_score or (abs(score - best_score) < 0.01 and avg_res < best_res):
                best_score, best_res = score, avg_res
                best = {"matched_frets": match, "offset": 0.0, "scale": scale,
                        "avg_residual_px": avg_res, "fret_start": 0,
                        "run_len": 0, "coverage_ratio": covered,
                        "n_visible": n_visible, "nut_prior": 1.0,
                        "score": score, "explained": explained}
    return best


def step8_fit_fret_rule(detected_x: list,
                        tol_px: float = CFG["step8_tol_px"],
                        ratio_tol: float = CFG["step8_ratio_tol"],
                        scale_min_factor: float = CFG["step8_scale_min_factor"],
                        scale_max_factor: float = CFG["step8_scale_max_factor"],
                        nut_anchored: bool = False,
                        nut_side: Optional[str] = None,
                        inlay_xs: list = None) -> dict:
    """17.817-es bund-illesztés (v11/v13)."""
    W = CFG["canonical_w"]
    empty = {
        "matched_frets": {}, "predicted_x": {},
        "offset": 0.0, "scale": float(W),
        "inlier_count": 0, "inlier_rate": 0.0,
        "avg_residual_px": float("nan"),
        "visible_range": (0, 0),
        "fit_method": "none", "run_len": 0,
        "fit_direction": "forward",
        "coverage_ratio": 0.0, "n_visible": 0, "score": 0.0,
        "nut_anchored": False,
        "inlay_score": 0.0, "matched_inlays": {},
    }
    if len(detected_x) < 2:
        return empty

    xs = np.array(sorted(detected_x))
    scale_min = (W / FRET_POS_NORM[N_FRETS]) * scale_min_factor
    scale_max = W * scale_max_factor

    if nut_anchored and nut_side in ("left", "right"):
        _direction = "forward" if nut_side == "left" else "reversed"
    else:
        _direction = "forward"
        if len(xs) >= 3:
            spacings = np.diff(xs)
            if len(spacings) >= 2:
                trend = float(np.polyfit(np.arange(len(spacings)), spacings, 1)[0])
                if trend > spacings.mean() * 0.05:
                    _direction = "reversed"

    xs_fit = xs if _direction == "forward" else np.array(sorted(float(W) - xs[::-1]))
    fit_direction = _direction
    best = None
    fit_method = "none"

    inlay_xs = inlay_xs or []
    if inlay_xs:
        inlay_fit = _fit_with_inlay_anchor(xs_fit, inlay_xs, nut_anchored,
                                           scale_min, scale_max, tol_px)
        if inlay_fit is not None:
            best = inlay_fit
            fit_method = "inlay_anchored"

    if best is None and nut_anchored:
        best = _fit_with_nut_anchor(xs_fit, tol_px, scale_min, scale_max)
        if best is not None:
            fit_method = "nut_anchored"
            print(f"  [fret_fit v11] nut-anchored | scale={best['scale']:.1f}px | "
                  f"cov={best['coverage_ratio']:.0%} ({len(best['matched_frets'])}/{best['n_visible']})")

    if best is None:
        best_score, best_res = 0.0, float("inf")
        runs = _ratio_runs(xs_fit, target=_TARGET_RATIO, tol=ratio_tol)
        for (r_start, r_end) in sorted(runs, key=lambda r: -(r[1] - r[0])):
            candidate = _fit_from_run(xs_fit, r_start, r_end, tol_px, scale_min, scale_max)
            if candidate is None:
                continue
            sc = candidate.get("score", 0.0)
            res = candidate["avg_residual_px"]
            if sc > best_score or (abs(sc - best_score) < 0.01 and res < best_res):
                best_score, best_res = sc, res
                best = candidate
        if best is not None:
            fit_method = "ratio_run"

    if best is None:
        print("  [fret_fit] Nincs konzisztens ratio-run → fallback RANSAC")
        best = _fit_constrained_ransac(xs_fit, tol_px, scale_min, scale_max)
        if best is not None:
            fit_method = "ransac_fallback"

    if best is None:
        print("  [fret_fit] Minden fit-módszer sikertelen.")
        return empty

    mf = best["matched_frets"]
    offset, scale = best["offset"], best["scale"]
    n_in = len(mf)

    pred_x = {}
    for n in range(N_FRETS + 1):
        xp = offset + scale * FRET_POS_NORM[n]
        if 0 <= xp <= W:
            pred_x[n] = float(xp)

    visible_range = (
        min(mf.keys(), default=0),
        max(mf.keys(), default=0),
    ) if mf else (0, 0)

    cov_ratio = best.get("coverage_ratio", 0.0)
    n_visible = best.get("n_visible", 0)
    score = best.get("score", 0.0)

    if fit_method != "nut_anchored":
        print(f"  [fret_fit] módszer: {fit_method} | irány: {fit_direction} | "
              f"offset={offset:.1f}px | scale={scale:.1f}px | "
              f"score={score:.2f} | cov={cov_ratio:.0%} ({n_in}/{n_visible}) | "
              f"res={best['avg_residual_px']:.2f}px | "
              f"látható: {visible_range[0]}–{visible_range[1]}")

    return {
        "matched_frets": mf,
        "predicted_x": pred_x,
        "offset": offset,
        "scale": scale,
        "inlier_count": n_in,
        "inlier_rate": float(n_in / max(len(xs), 1)),
        "avg_residual_px": best["avg_residual_px"],
        "visible_range": visible_range,
        "fit_method": fit_method,
        "run_len": best.get("run_len", 0),
        "inlay_score": best.get("inlay_score", 0.0),
        "matched_inlays": best.get("matched_inlays", {}),
        "fit_direction": fit_direction,
        "coverage_ratio": cov_ratio,
        "n_visible": n_visible,
        "score": score,
        "nut_anchored": fit_method == "nut_anchored",
    }
