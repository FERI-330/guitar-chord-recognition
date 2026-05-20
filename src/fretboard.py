"""
src/fretboard.py

Teljes run_v14_pipeline orchestrátor + validálók + suppress.

Forrás: 03c_pipeline_fixes_design.ipynb (V14), cellák 23, 25, 27.

Plug-and-Play bunddetektáló architektúra:
  FretDetectorInterface  – ABC, közös interfész
    GeometricFretDetector  – meglévő Hough+step8 logika (fallback)
    IntensityFretDetector  – Sobel-X gradiens csúcsdetektálás + step8 (default)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
from typing import Optional

import cv2
import numpy as np

from src.config import CFG, FRET_ENGINE, FRET_ENGINE_FALLBACK
from src.constants import CANONICAL_W, CANONICAL_H
from src.geometry import (
    bgr2rgb, load_image_bgr,
    step1_canny, step2_hough, step3_neck_angle, step3b_refine_neck_angle,
    step4_split_lines,
    step5_outer_edges, step6_clamp_trapezoid_extent, step6_trapezoid, step6_warp,
    step6d_shear_correction,
    step7_fret_lines_canonical, step8_fit_fret_rule,
    detect_guitar_orientation,
)
# step6b_find_nut / step6c_trim_to_nut / step6_extend_for_nut → prototype_nut_detector.py
from src.hand_landmark import (
    get_landmarker,
    step9_detect_landmarks, step9_project_fingertips,
    build_finger_mask, anchor_neck_angle, step3_neck_angle_anchored,
)
# get_fretboard_near_edge → prototype_nut_detector.py (only needed for nut boundary search)

# Modul-szintű lazy singleton a landmarker-hez
_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        _landmarker = get_landmarker()
    return _landmarker


@contextlib.contextmanager
def _config_patch(**overrides):
    """Temporarily override CFG values inside the pipeline."""
    cfg_backup = {}
    for key, value in overrides.items():
        if key in CFG:
            cfg_backup[key] = CFG[key]
            CFG[key] = value
    try:
        yield
    finally:
        CFG.update(cfg_backup)


def _make_debug_info(stage: str, exc: Exception | str, **extra) -> dict:
    """Compact diagnostic payload for partial pipeline failures."""
    info = {"stage": stage, "error": str(exc)}
    for key, value in extra.items():
        if value is not None:
            info[key] = value
    return info


def _derive_is_flipped(fit: Optional[dict],
                       orientation: Optional[dict],
                       landmarks: Optional[list]) -> bool:
    """Determines if the guitar is 'flipped' (nut on the right side of canonical image).

    Primary evidence: fret spacing gradient from fit["fit_direction"].
      fit_direction="reversed" → spacings increase left→right → nut is on the right.
    Fallback: wrist-vs-index_mcp orientation from detect_guitar_orientation.
    Last resort: raw landmark direction.

    Returns True when the canonical image needs horizontal mirroring for standard
    nut-left normalization.
    """
    # Primary: fit direction from fret spacing gradient (reliable when coverage ≥ 0.3)
    if fit is not None:
        direction = fit.get("fit_direction")
        if direction in ("forward", "reversed") and float(fit.get("coverage_ratio", 0.0)) >= 0.30:
            return direction == "reversed"

    # Fallback: landmark-based orientation (flip_logic = side_hint == "right")
    if orientation is not None:
        return bool(orientation.get("flip_logic", False))

    # Last resort: raw wrist vs index_mcp x-position
    if landmarks is not None and len(landmarks) >= 6:
        wrist_x = float(landmarks[0][0])
        index_mcp_x = float(landmarks[5][0])
        if abs(index_mcp_x - wrist_x) > 1e-6:
            return index_mcp_x < wrist_x  # wrist further right → nut on right

    return False


def _global_hough_fallback(img: np.ndarray, edges: np.ndarray) -> list:
    """Permissive HoughLinesP when standard step2_hough finds nothing (no hand).

    Finds near-horizontal lines (|angle| ≤ 15°) sorted by length descending.
    Returns same format as step2_hough: list of (x1, y1, x2, y2) tuples.
    """
    h, w = edges.shape[:2]
    min_len = max(w // 5, 60)
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=20, minLineLength=min_len, maxLineGap=30,
    )
    if raw is None:
        return []
    result = []
    for ln in raw:
        x1, y1, x2, y2 = int(ln[0][0]), int(ln[0][1]), int(ln[0][2]), int(ln[0][3])
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if angle > 90.0:
            angle = 180.0 - angle
        if angle <= 15.0:
            length = float(np.hypot(x2 - x1, y2 - y1))
            result.append((length, (x1, y1, x2, y2)))
    result.sort(key=lambda p: -p[0])
    return [p[1] for p in result]


# ─────────────────────────────────────────────────────────────────────────────
# Trapézoid validálás
# ─────────────────────────────────────────────────────────────────────────────

def validate_trapezoid(corners: np.ndarray,
                       img_shape: tuple,
                       landmarks: Optional[list] = None,
                       min_aspect: float = 1.2,
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

    min_aspect = float(CFG.get("sanity_min_aspect", min_aspect))
    area_limits = CFG.get("sanity_area_limits", {}) or {}
    area_frac_range = (
        float(area_limits.get("min_frac", area_frac_range[0])),
        float(area_limits.get("max_frac", area_frac_range[1])),
    )
    max_edge_angle_diff_deg = float(CFG.get("sanity_max_edge_angle_diff_deg", max_edge_angle_diff_deg))

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
                     img_shape: tuple,
                     landmarks: Optional[list] = None) -> Optional[str]:
    """A nut oldalának meghatározása a kanonikus képen.

    Elsődleges módszer (robusztus magas fogásállásban is): az ujjhegyek
    kanonikus x átlagát a csukló kanonikus x-éhez hasonlítja. Az ujjhegyek
    a nut felé mutatnak → ha mean(tip_cx) < wrist_cx, a nut bal oldalt van.

    Fallback: tenyér-centrum 50%-os küszöb (csak ha landmarks nem elérhető).
    """
    if anchor is None or H is None:
        return None

    h_img, w_img = img_shape[:2]

    # ── Elsődleges: wrist.x vs index_mcp.x kép-koordinátában ────────────────
    # Ha a csukló x > mutatóujj tő x → a kéz balra mutat → nut bal oldalt
    # Közvetlenül a normalizált landmark koordinátákon működik, vetítés nélkül.
    if landmarks is not None and len(landmarks) >= 6:
        wrist_x     = float(landmarks[0][0])   # lm[0] = wrist
        index_mcp_x = float(landmarks[5][0])   # lm[5] = index_mcp
        side = "left" if wrist_x > index_mcp_x else "right"
        print(f"  [nut_side] wrist.x={wrist_x:.3f} vs index_mcp.x={index_mcp_x:.3f} → {side}")
        return side

    # ── 2. Fallback: ujjhegy-irány a kanonikus térben ────────────────────────
    if landmarks is not None and len(landmarks) >= 21:
        wrist_lm = landmarks[0]
        wrist_pt = np.array([wrist_lm[0] * w_img, wrist_lm[1] * h_img, 1.0])
        wrist_proj = H @ wrist_pt
        if abs(wrist_proj[2]) > 1e-9:
            wrist_cx = float(wrist_proj[0] / wrist_proj[2])
            tip_cxs = []
            for tip_idx in (4, 8, 12, 16, 20):
                xn, yn, _ = landmarks[tip_idx]
                pt = np.array([xn * w_img, yn * h_img, 1.0])
                proj = H @ pt
                if abs(proj[2]) > 1e-9:
                    tip_cxs.append(float(proj[0] / proj[2]))
            if tip_cxs:
                mean_tip_cx = float(np.mean(tip_cxs))
                side = "left" if mean_tip_cx < wrist_cx else "right"
                print(f"  [nut_side] tip_mean={mean_tip_cx:.0f} vs wrist={wrist_cx:.0f} → {side}")
                return side

    # ── 3. Végső fallback: tenyér-centrum 50%-os küszöb ─────────────────────
    pc = anchor["palm_center_px"]
    pt = np.array([float(pc[0]), float(pc[1]), 1.0])
    proj = H @ pt
    if abs(proj[2]) < 1e-9:
        return None
    cx = float(proj[0] / proj[2])
    return "left" if cx < CANONICAL_W / 2.0 else "right"


# ─────────────────────────────────────────────────────────────────────────────
# Landmark → kanonikus tér vetítés
# ─────────────────────────────────────────────────────────────────────────────

def _project_landmark_to_canon(lm_px: tuple[float, float],
                               H: np.ndarray) -> Optional[float]:
    """Egy pixel-koordinátát H homográfián vetít a kanonikus térbe.

    Visszaad: kanonikus x-koordináta (float), vagy None ha H degenerate.
    """
    pt = np.array([lm_px[0], lm_px[1], 1.0])
    proj = H @ pt
    if abs(proj[2]) < 1e-9:
        return None
    return float(proj[0] / proj[2])


# ─────────────────────────────────────────────────────────────────────────────
# Post-fit fantom-bund szűrés
# ─────────────────────────────────────────────────────────────────────────────

def refine_frets_by_fit(fret_xs: list,
                        fit: dict,
                        tol_px: float) -> list:
    """Csak azokat a fret_xs pozíciókat tartja meg, amelyek az előrejelzett
    bundpozíció ±tol_px sugarán belül vannak.

    Ha a fit üres (nincs predicted_x), az eredeti listát adja vissza.
    Így a függvény mindig biztonságos fallback-ként viselkedik.
    """
    pred = fit.get("predicted_x", {})
    if not pred or not fret_xs:
        return list(fret_xs)
    pred_vals = list(pred.values())
    kept = []
    for x in fret_xs:
        nearest_dist = min(abs(x - px) for px in pred_vals)
        if nearest_dist <= tol_px:
            kept.append(x)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Bunddetektáló architektúra – FretDetectorInterface + implementációk
# ─────────────────────────────────────────────────────────────────────────────

def _make_empty_fit() -> dict:
    """Üres fit dict – azonos struktúrával mint step8_fit_fret_rule failure."""
    return {
        "matched_frets": {}, "predicted_x": {},
        "offset": 0.0, "scale": float(CFG["canonical_w"]),
        "inlier_count": 0, "inlier_rate": 0.0,
        "avg_residual_px": float("nan"),
        "visible_range": (0, 0),
        "fit_method": "none", "run_len": 0,
        "fit_direction": "forward",
        "coverage_ratio": 0.0, "n_visible": 0, "score": 0.0,
        "nut_anchored": False,
        "inlay_score": 0.0, "matched_inlays": {},
        "method": "none",
    }


class FretDetectorInterface(ABC):
    """Közös interfész minden bunddetektáló implementációhoz.

    A "Zero-Break" garancia alapja: ``detect()`` mindig ugyanazokat a kulcsokat
    adja vissza, függetlenül a konkrét algoritmustól.  Így ``run_v14_pipeline``
    és ``assemble_feature_vector`` nem veszi észre a cserét.

    Kötelező visszatérési kulcsok:
        ``fit``            – dict, azonos struktúra mint step8_fit_fret_rule output
        ``fret_xs_raw``    – list[float], nyers detektált x-pozíciók
        ``fret_xs_filt``   – list[float], szűrt pozíciók
        ``removed_pairs``  – list[tuple], eltávolított párok
        ``method``         – str, 'geometric' | 'intensity'
    """

    @abstractmethod
    def detect(self, canon_bgr: np.ndarray,
               nut: Optional[dict] = None,
               shear: Optional[dict] = None,
               hand_mask: Optional[np.ndarray] = None) -> dict:
        """Bundpozíciók detektálása a kanonikus képen.

        Args:
            canon_bgr: kanonikus perspektíva-warped BGR kép (600×80 px).
            nut:       nut detektálás eredménye (``step6b_find_nut`` kimenete),
                       vagy ``None`` ha nem elérhető.
            shear:     step6d_shear_correction kimenete – az auto-mode dönti el
                       Sobel-X vs Max-pooling választást.
            hand_mask: opcionális kanonikus kézmaszk (uint8) – az ujjak
                       területe elnyomható a bund-detektálás során.

        Returns:
            Dict kötelező kulcsokkal: fit, fret_xs_raw, fret_xs_filt,
            removed_pairs, method.
        """


class GeometricFretDetector(FretDetectorInterface):
    """Eredeti V14 Hough-alapú bunddetektálás (step7 + suppress + step8).

    Ez az osztály kizárólag átcsomagolja a meglévő ``geometry.py`` függvényeket –
    semmi logikát nem implementál újra.  Alapértelmezett detektor a pipeline-ban.
    """

    def detect(self, canon_bgr: np.ndarray,
               nut: Optional[dict] = None,
               shear: Optional[dict] = None,
               hand_mask: Optional[np.ndarray] = None) -> dict:
        try:
            fret_xs_raw = step7_fret_lines_canonical(canon_bgr)
            fret_xs_filt, removed_pairs = suppress_finger_pairs(fret_xs_raw)

            refine_enabled = bool(CFG.get("fret_refine_enabled", True))
            refine_tol = float(CFG.get("fret_refine_tol_px", 12.0))

            # Nut-anchor eltávolítva a kritikus útból – lebegő bund-hálózat
            fit_pass1 = step8_fit_fret_rule(fret_xs_filt, nut_anchored=False)
            if refine_enabled and len(fret_xs_filt) >= 3:
                refined_frets = refine_frets_by_fit(fret_xs_filt, fit_pass1, refine_tol)
                if refined_frets:
                    fret_xs_filt = refined_frets
                fit = step8_fit_fret_rule(fret_xs_filt, nut_anchored=False)
            else:
                fit = fit_pass1
        except Exception as exc:
            print(f"  [GeometricFretDetector] hiba: {exc}")
            debug_info = _make_debug_info("geometric_detect", exc)
            return {
                "fit": _make_empty_fit(),
                "fret_xs_raw": [],
                "fret_xs_filt": [],
                "removed_pairs": [],
                "method": "geometric",
                "debug_info": debug_info,
            }

        fit["method"] = "geometric"
        result = {
            "fit":           fit,
            "fret_xs_raw":   fret_xs_raw,
            "fret_xs_filt":  fret_xs_filt,
            "removed_pairs": removed_pairs,
            "method":        "geometric",
        }
        if fit_pass1.get("score", 0.0) <= 0.0 and not fret_xs_filt:
            result["debug_info"] = _make_debug_info("geometric_detect", "no_frets_found")
        return result


class IntensityFretDetector(FretDetectorInterface):
    """Intenzitás-alapú bunddetektálás konfigurálható profilstratégiával + step8.

    mode="sobel"  – |Sobel-X|.sum(axis=0)  →  maximális precizitás egyenes bundoknál
    mode="max"    – np.max(gray, axis=0)   →  robusztus dőlt/zajos képeknél
    mode="auto"   – shear-eredmény alapján automatikusan választ:
                    ha a shear-korrekció sikeres volt (corrected=True) VAGY
                    az egyenes bund bizonyítható (n_lines≥4 és |α|<0.2°),
                    akkor Sobel-X-et használ; egyébként Max-pooling-ot.

    A 17.817-es illesztést minden módban ``step8_fit_fret_rule`` végzi.
    """

    # SNR-küszöb az auto fallback döntésnél: ha Sobel SNR < ennél, Max-pooling-ra vált
    _SNR_FALLBACK_THR: float = 1.5

    def __init__(
        self,
        mode: str = "auto",
        sobel_ksize: int = 3,
        smooth_sigma: float = 1.5,
        peak_height: float = 0.12,
        peak_distance: int = 7,
        peak_prominence: float = 0.06,
        peak_max_width: float = 14.0,
        suppress_pairs: bool = True,
        power: float = 2.0,
    ) -> None:
        self.mode            = str(mode).lower()
        self.sobel_ksize     = sobel_ksize
        self.smooth_sigma    = smooth_sigma
        self.peak_height     = peak_height
        self.peak_distance   = peak_distance
        self.peak_prominence = peak_prominence
        self.peak_max_width  = peak_max_width
        self.suppress_pairs  = suppress_pairs
        self.power           = power

    # ── Belső segédmetódusok ──────────────────────────────────────────────────

    def _norm_smooth(self, raw: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter1d
        if self.smooth_sigma > 0:
            raw = gaussian_filter1d(raw, sigma=self.smooth_sigma)
        mx = float(raw.max())
        return (raw / mx).astype(np.float32) if mx > 1e-3 else raw.astype(np.float32)

    def _sobel_profile(self, gray: np.ndarray) -> np.ndarray:
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.sobel_ksize)
        return self._norm_smooth(np.abs(sx).sum(axis=0))

    def _max_profile(self, gray: np.ndarray) -> np.ndarray:
        """Oszloponkénti maximum – dőlésre-invariáns, robusztus."""
        return self._norm_smooth(gray.max(axis=0).astype(np.float32))

    def _linear_profile(self, gray: np.ndarray) -> np.ndarray:
        """Oszloponkénti átlag – baseline referencia."""
        return self._norm_smooth(gray.mean(axis=0))

    def _power_profile(self, gray: np.ndarray) -> np.ndarray:
        """Power transform: mean((gray/255)^power) – csúcserősítés."""
        return self._norm_smooth(((gray / 255.0) ** self.power).mean(axis=0))

    def _dispatch_profile(self, gray: np.ndarray, mode: str) -> np.ndarray:
        """Profil mód szerinti dispatch (linear | power | max | sobel)."""
        if mode == "linear":
            return self._linear_profile(gray)
        if mode == "power":
            return self._power_profile(gray)
        if mode == "max":
            return self._max_profile(gray)
        return self._sobel_profile(gray)

    def _snr(self, profile: np.ndarray, thr: float = 0.30) -> float:
        above = profile[profile >= thr]
        below = profile[profile <  thr]
        if len(above) == 0 or len(below) == 0:
            return 0.0
        return float(above.mean() / (below.mean() + 1e-9))

    def _select_mode(self, shear: Optional[dict]) -> tuple[str, dict]:
        """Shear-eredmény alapján Sobel-X vagy Max-pooling választása.

        Sobel-X (precizitás): residual < 0.3° és magas Hough-bizalom.
        Max-pooling (robusztus): minden más esetben, beleértve a kevés vonalat.
        """
        if shear is None:
            return "max", {
                "strategy": "max",
                "reason": "no_shear_info",
                "residual_tilt_deg": None,
                "hough_confidence": None,
                "n_lines": 0,
            }
        residual = abs(float(shear.get("residual_shear_deg", shear.get("shear_angle_deg", 0.0))))
        confidence = float(shear.get("hough_confidence", 0.0))
        n_lines = int(shear.get("n_lines", 0))
        if n_lines >= 4 and residual < 0.3 and confidence >= 0.75:
            return "sobel", {
                "strategy": "sobel",
                "reason": "low_residual_high_confidence",
                "residual_tilt_deg": residual,
                "hough_confidence": confidence,
                "n_lines": n_lines,
            }
        return "max", {
            "strategy": "max",
            "reason": "high_tilt_or_low_confidence",
            "residual_tilt_deg": residual,
            "hough_confidence": confidence,
            "n_lines": n_lines,
        }

    # ── Nyilvános API ─────────────────────────────────────────────────────────

    def gradient_profile(self, canon_bgr: np.ndarray,
                         shear: Optional[dict] = None) -> np.ndarray:
        """Normalizált 1D profil a konfigurált módban.

        Nyilvános metódus: vizualizálható a PipelineVisualizer-ből.
        """
        gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        active = self.mode if self.mode != "auto" else self._select_mode(shear)[0]
        return self._dispatch_profile(gray, active)

    def detect(self, canon_bgr: np.ndarray,
               nut: Optional[dict] = None,
               shear: Optional[dict] = None,
               hand_mask: Optional[np.ndarray] = None) -> dict:
        from scipy.signal import find_peaks

        debug_info = {}
        profile = None
        raw_profile = None
        active_mode = self.mode
        auto_meta = None
        try:
            gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            raw_profile = self._linear_profile(gray)
            if self.mode == "auto":
                active_mode, auto_meta = self._select_mode(shear)
            profile = self._dispatch_profile(gray, active_mode)
            profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
            raw_profile = np.nan_to_num(raw_profile, nan=0.0, posinf=0.0, neginf=0.0)

            # Kézmaszk alapú elnyomás: ujj-területeken az intenzitás-profil 0-ra áll,
            # hogy ott ne kerüljön bund-jelölt detektálásra. A vizuális canon kép érintetlen.
            if hand_mask is not None:
                col_has_hand = np.any(hand_mask > 0, axis=0)
                if col_has_hand.shape[0] == profile.shape[0]:
                    profile[col_has_hand] = 0.0

            # SNR-alapú fallback csak auto+sobel esetén: ha Sobel gyenge → Max-pooling
            if active_mode == "sobel" and self.mode == "auto":
                snr_val = self._snr(profile)
                if snr_val < self._SNR_FALLBACK_THR:
                    profile = self._max_profile(gray)
                    active_mode = "max"
                    if auto_meta is not None:
                        auto_meta = {
                            **auto_meta,
                            "strategy": "max",
                            "reason": "sobel_low_snr_fallback",
                            "sobel_snr": snr_val,
                        }
                    print(f"  [IntensityFretDetector] Sobel SNR={snr_val:.2f} → Max-pooling fallback")

            residual = None if shear is None else float(shear.get("residual_shear_deg", shear.get("shear_angle_deg", 0.0)))
            confidence = None if shear is None else float(shear.get("hough_confidence", 0.0))
            _shear_info = "n/a" if shear is None else f"res={residual:.2f}° conf={confidence:.2f}"
            print(f"  [IntensityFretDetector] mode={active_mode} | shear={_shear_info}")

            residual = abs(float(shear.get("residual_shear_deg", shear.get("shear_angle_deg", 0.0)))) if shear else 0.0
            conf = float(shear.get("hough_confidence", 0.0)) if shear else 0.0
            dyn_prom = float(np.clip(
                self.peak_prominence * (1.0 + residual / 0.6) * (1.0 + max(0.0, 0.7 - conf) * 0.5),
                0.02,
                0.35,
            ))
            min_width = 1.0 if active_mode == "sobel" else 2.0
            max_width = float(np.clip(
                self.peak_max_width + (3.0 if active_mode == "max" else 0.0) + min(residual * 4.0, 4.0),
                min_width + 0.5,
                28.0,
            ))

            peak_idxs, _ = find_peaks(
                profile,
                height=self.peak_height,
                distance=self.peak_distance,
                prominence=dyn_prom,
                width=(min_width, max_width),
            )

            if len(peak_idxs) < 2:
                relaxed_idxs, _ = find_peaks(
                    profile,
                    height=max(0.02, self.peak_height * 0.75),
                    distance=max(3, self.peak_distance // 2),
                    prominence=max(0.02, dyn_prom * 0.5),
                    width=(1.0, None),
                )
                if len(relaxed_idxs) > len(peak_idxs):
                    debug_info["peak_fallback"] = "relaxed_find_peaks"
                    peak_idxs = relaxed_idxs

            fret_xs_raw = [float(i) for i in peak_idxs]
            if len(fret_xs_raw) < 2:
                raise RuntimeError("too_few_peak_candidates")

            if self.suppress_pairs:
                fret_xs_filt, removed_pairs = suppress_finger_pairs(fret_xs_raw)
            else:
                fret_xs_filt, removed_pairs = fret_xs_raw, []

        except Exception as exc:
            debug_info.update(_make_debug_info("peak_detection", exc, mode=active_mode))
            try:
                fret_xs_raw = step7_fret_lines_canonical(canon_bgr)
                fret_xs_filt, removed_pairs = suppress_finger_pairs(fret_xs_raw) if self.suppress_pairs else (list(fret_xs_raw), [])
                debug_info["fallback"] = "geometric_step7"
            except Exception as fallback_exc:
                debug_info["fallback_error"] = str(fallback_exc)
                fret_xs_raw = []
                fret_xs_filt = []
                removed_pairs = []
                profile = None
                raw_profile = None

        # Intensity detector: do not require a detected Nut for fitting.
        # Treat any provided `nut` only as debug information, but do not
        # anchor the fit to it. This makes detection robust when Nut is missing.
        nut_side = None
        if nut:
            debug_info["nut_provided"] = True
            debug_info["nut_info"] = {k: nut.get(k) for k in ("side", "nut_x", "width_px") if k in nut}
        refine_enabled = bool(CFG.get("fret_refine_enabled", True))
        refine_tol = float(CFG.get("fret_refine_tol_px", 12.0))

        try:
            # Do not anchor the fit to Nut for intensity-based detector.
            fit_pass1 = step8_fit_fret_rule(
                fret_xs_filt,
                nut_anchored=False,
                nut_side=None,
            )
            fit = fit_pass1
            if refine_enabled and len(fret_xs_filt) >= 3:
                refined_frets = refine_frets_by_fit(fret_xs_filt, fit_pass1, refine_tol)
                if refined_frets:
                    fret_xs_filt = refined_frets
                    fit = step8_fit_fret_rule(
                        fret_xs_filt,
                        nut_anchored=False,
                        nut_side=None,
                    )
        except Exception as exc:
            debug_info.update(_make_debug_info("step8_fit", exc, mode=active_mode))
            fit = _make_empty_fit()

        fit["method"] = f"intensity_{active_mode}"
        if auto_meta is not None:
            auto_meta = {
                **auto_meta,
                "active_mode": active_mode,
                "peak_prominence": locals().get("dyn_prom"),
                "peak_width_min": locals().get("min_width"),
                "peak_width_max": locals().get("max_width"),
            }

        result = {
            "fit":           fit,
            "fret_xs_raw":   fret_xs_raw,
            "fret_xs_filt":  fret_xs_filt,
            "removed_pairs": removed_pairs,
            "method":        f"intensity_{active_mode}",
            "profile":       profile,
            "profile_raw":   raw_profile,
            "profile_mode":  active_mode,
            "auto_strategy": auto_meta,
        }
        if debug_info:
            result["debug_info"] = debug_info
        return result


_FRET_DETECTOR_FACTORIES = {
    FRET_ENGINE_FALLBACK: GeometricFretDetector,
    FRET_ENGINE: IntensityFretDetector,
}


def _make_default_fret_detector() -> FretDetectorInterface:
    """A CFG-ben beállított alapértelmezett bunddetektor példányosítása."""
    engine = str(CFG.get("fret_engine", FRET_ENGINE)).upper()
    detector_cls = _FRET_DETECTOR_FACTORIES.get(engine, IntensityFretDetector)
    return detector_cls()


# ─────────────────────────────────────────────────────────────────────────────
# run_v14_pipeline – a fő orchestrátor
# ─────────────────────────────────────────────────────────────────────────────

def run_v14_pipeline(img_entry: dict,
                     landmarker=None,
                     fret_detector: Optional[FretDetectorInterface] = None,
                     preprocessor=None) -> dict:
    """Egy képre lefuttatja a V14 pipeline-t.

    Args:
        img_entry:      dict {'path', 'class', ...} – általában manifest sor.
        landmarker:     HandLandmarker vagy None (ilyenkor lazy singleton).
        fret_detector:  ``FretDetectorInterface`` példány, vagy ``None``.
                        ``None`` esetén a ``CFG['fret_engine']`` alapján
                        példányosított motor (alapból ``INTENSITY_DATA``).

    Visszaad: dict minden közbülső artefaktummal + 'ok' flag.
    Plusz kulcs az alap result dicthez: 'fret_detector_method' str.
    """
    if landmarker is None:
        landmarker = _get_landmarker()

    out = {
        "class": img_entry.get("class", "?"),
        "path": img_entry["path"],
        "fname": img_entry.get("fname", str(img_entry["path"]).split("/")[-1]),
        "ok": False,
        "invalid_reason": None,
        "fret_detector_method": "none",
        "intensity_profile_mode": None,
        "intensity_auto_strategy": None,
        "debug_info": {},
    }

    try:
        img = load_image_bgr(img_entry["path"])
    except FileNotFoundError as exc:
        out["invalid_reason"] = "load_failed"
        out["debug_info"] = _make_debug_info("load_image", exc)
        return out
    out["img"] = img

    landmarks = None
    try:
        # Landmarks detektálása az eredeti képen fut — CLAHE ronthatja a MediaPipe teljesítményét.
        landmarks = step9_detect_landmarks(img_entry["path"], landmarker)
    except Exception as exc:
        out["debug_info"]["landmarks"] = _make_debug_info("landmarks", exc)
    out["landmarks"] = landmarks

    try:
        anchor = anchor_neck_angle(landmarks, img.shape)
    except Exception as exc:
        anchor = None
        out["debug_info"]["anchor"] = _make_debug_info("anchor", exc)
    out["anchor"] = anchor

    try:
        orientation = detect_guitar_orientation({"landmarks": landmarks, "img_shape": img.shape})
    except Exception as exc:
        orientation = None
        out["debug_info"]["orientation"] = _make_debug_info("orientation", exc)
    out["guitar_orientation"] = orientation

    try:
        finger_mask = build_finger_mask(img.shape, landmarks)
    except Exception as exc:
        finger_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        out["debug_info"]["finger_mask"] = _make_debug_info("finger_mask", exc)
    out["finger_mask"] = finger_mask

    if preprocessor is not None:
        try:
            img = preprocessor.process(img)
            out["img_preprocessed"] = img
        except Exception as exc:
            out["debug_info"]["preprocess"] = _make_debug_info("preprocess", exc)

    try:
        edges = step1_canny(img)
        edges_masked = edges.copy()
        if finger_mask.any():
            edges_masked[finger_mask > 0] = 0
        out["edges"] = edges
        out["edges_masked"] = edges_masked
    except Exception as exc:
        out["invalid_reason"] = f"canny_failed: {exc}"
        out["debug_info"] = {**out["debug_info"], "edges": _make_debug_info("edges", exc)}
        return out

    try:
        lines = step2_hough(img, edges_masked)
        out["lines"] = lines
    except Exception as exc:
        out["invalid_reason"] = f"hough_failed: {exc}"
        out["debug_info"] = {**out["debug_info"], "hough": _make_debug_info("hough", exc)}
        return out
    if not lines:
        if landmarks is None:
            # No hand detected → try permissive global search for near-horizontal neck edges
            lines = _global_hough_fallback(img, edges)
            if lines:
                out["debug_info"]["hough"] = {"fallback": "global_hough_no_hand", "n_lines": len(lines)}
                print(f"  [no_hand_fallback] {len(lines)} near-horizontal line(s) via global Hough")
            else:
                out["invalid_reason"] = "no_hough_lines_no_hand"
                out["debug_info"]["hough"] = _make_debug_info("hough", "no_hough_lines_no_hand")
                return out
        else:
            out["invalid_reason"] = "no_hough_lines"
            out["debug_info"]["hough"] = _make_debug_info("hough", "no_hough_lines")
            return out

    try:
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

        if len(split["fret_lines"]) >= 3:
            refined_angle = step3b_refine_neck_angle(neck["angle_deg"], split["fret_lines"])
            if abs(refined_angle - neck["angle_deg"]) > 0.1:
                split = step4_split_lines(lines, refined_angle)
                neck = dict(neck)
                neck["angle_deg"] = refined_angle
                neck["angle_refined"] = True

        out["neck"] = neck
        out["split"] = split
    except Exception as exc:
        out["invalid_reason"] = f"neck_split_failed: {exc}"
        out["debug_info"]["neck"] = _make_debug_info("neck_split", exc)
        return out

    if not split["long_lines"]:
        out["invalid_reason"] = "no_long_lines"
        out["debug_info"]["neck"] = _make_debug_info("neck_split", "no_long_lines")
        return out

    try:
        edge_info = step5_outer_edges(split["long_lines"], neck["angle_deg"])
        out["edge_info"] = edge_info
    except Exception as exc:
        out["invalid_reason"] = f"outer_edges_failed: {exc}"
        out["debug_info"]["outer_edges"] = _make_debug_info("outer_edges", exc)
        return out
    if edge_info is None:
        out["invalid_reason"] = "no_outer_edges"
        out["debug_info"]["outer_edges"] = _make_debug_info("outer_edges", "no_outer_edges")
        return out

    try:
        edge_info = step6_clamp_trapezoid_extent(edge_info, anchor)
    except Exception as exc:
        out["debug_info"]["clamp_extent"] = _make_debug_info("clamp_extent", exc)

    trap_overrides = {}
    if orientation is not None:
        trap_overrides["nut_extend_amin_margin_px"] = int(orientation.get(
            "extend_margin_px", CFG.get("nut_extend_amin_margin_px", 120)
        ))
    try:
        with _config_patch(**trap_overrides):
            trap = step6_trapezoid(img, edge_info, landmarks=landmarks)
    except Exception as exc:
        trap = None
        out["invalid_reason"] = f"trapezoid_failed: {exc}"
        out["debug_info"]["trapezoid"] = _make_debug_info("trapezoid", exc)
    out["trap"] = trap
    if trap is None:
        if out["invalid_reason"] is None:
            out["invalid_reason"] = "no_trapezoid"
        return out

    # ROI minimum height: if fretboard strip is thinner than 15% of the image height,
    # expand corners symmetrically outward along the perp direction.
    try:
        roi_height = min(trap["w_start"], trap["w_end"])
        min_h_px = img.shape[0] * float(CFG.get("roi_min_height_frac", 0.15))
        if roi_height < min_h_px:
            perp = (np.array(edge_info["perp_dir"], dtype=np.float64)
                    if edge_info is not None else np.array([0.0, 1.0]))
            expand = (min_h_px - roi_height) / 2.0
            corners = trap["corners_px"].astype(np.float64)
            projs = [float(np.dot(corners[i], perp)) for i in range(4)]
            left_idxs = sorted(range(4), key=lambda i: projs[i])[:2]
            right_idxs = sorted(range(4), key=lambda i: projs[i])[2:]
            for i in left_idxs:
                corners[i] -= perp * expand
            for i in right_idxs:
                corners[i] += perp * expand
            trap = dict(trap)
            trap["corners_px"] = np.clip(corners, 0, None).astype(np.float32)
            trap["w_start"] = float(np.linalg.norm(trap["corners_px"][1] - trap["corners_px"][0]))
            trap["w_end"] = float(np.linalg.norm(trap["corners_px"][2] - trap["corners_px"][3]))
            out["trap"] = trap
            new_h = min(trap["w_start"], trap["w_end"])
            out["debug_info"]["roi_min_height_expanded"] = {"from_px": roi_height, "to_px": new_h}
            print(f"  [roi_min_height] {roi_height:.1f}→{new_h:.1f}px "
                  f"(img_h={img.shape[0]}, thr={min_h_px:.0f}px)")
    except Exception as exc:
        out["debug_info"]["roi_min_height"] = _make_debug_info("roi_min_height", exc)

    try:
        ok, reasons = validate_trapezoid(trap["corners_px"], img.shape, landmarks)
    except Exception as exc:
        ok, reasons = False, [f"validate_failed: {exc}"]
        out["debug_info"]["trap_sanity"] = _make_debug_info("trap_sanity", exc)
    out["trap_ok"] = ok
    out["trap_reasons"] = reasons
    if not ok:
        out["debug_info"]["trap_sanity_warning"] = reasons
        print(f"  [trap_sanity] WARNING — continuing with sub-optimal trapezoid: {', '.join(reasons)}")

    try:
        H, H_inv, canon = step6_warp(img, trap["corners_px"])
    except Exception as exc:
        out["invalid_reason"] = f"warp_failed: {exc}"
        out["debug_info"]["warp"] = _make_debug_info("warp", exc)
        return out
    out["H"], out["H_inv"], out["canon"] = H, H_inv, canon

# Warp the original finger_mask into canonical ROI using the same homography.
# Do this separately (do not mask the source image before warping).
    try:
        fm = out.get("finger_mask")
        if fm is not None:
            Wc = int(CFG.get("canonical_w"))
            Hc = int(CFG.get("canonical_h"))
            try:
                hand_mask_canon = cv2.warpPerspective(fm, H, (Wc, Hc), flags=cv2.INTER_NEAREST)
            except Exception:
                # fallback: attempt simple resize
                hand_mask_canon = cv2.resize(fm, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
            # store canonical hand mask for downstream viz and feature extraction
            out["hand_mask"] = hand_mask_canon
    except Exception as exc:
        out["debug_info"]["hand_mask_warp"] = _make_debug_info("hand_mask_warp", exc)

    side_hint = None
    try:
        side_hint = orientation["side_hint"] if orientation and orientation.get("side_hint") else _choose_nut_side(anchor, H, img.shape, landmarks=landmarks)
    except Exception as exc:
        out["debug_info"]["nut_side_hint"] = _make_debug_info("nut_side_hint", exc)
    out["nut_side_hint"] = side_hint

    # Nut detekció ki van vezetve a kritikus útból → prototype_nut_detector.py kezeli (csak vizualizációhoz)
    out["nut"] = None

    out["canon_pre_shear"] = out.get("canon")
    try:
        shear = step6d_shear_correction(out["canon"])
    except Exception as exc:
        shear = {
            "corrected": False,
            "shear_angle_deg": 0.0,
            "residual_shear_deg": 0.0,
            "n_lines": 0,
            "hough_confidence": 0.0,
            "canon_corrected": out["canon"],
            "S": np.eye(3, dtype=np.float32),
            "S_inv": np.eye(3, dtype=np.float32),
        }
        out["debug_info"]["shear"] = _make_debug_info("shear", exc)
    out["shear"] = shear
    if shear.get("corrected"):
        out["H"]     = shear["S"] @ out["H"]
        out["H_inv"] = out["H_inv"] @ shear["S_inv"]
        out["canon"] = shear["canon_corrected"]
        # Hand mask shear-korrekciója: ugyanaz az affin transzformáció, mint a canon képen
        if out.get("hand_mask") is not None:
            try:
                Wc = int(CFG["canonical_w"])
                Hc = int(CFG["canonical_h"])
                S_mat = np.asarray(shear["S"], dtype=np.float64)
                M_affine_hm = S_mat[:2, :].astype(np.float32)
                out["hand_mask"] = cv2.warpAffine(
                    out["hand_mask"], M_affine_hm, (Wc, Hc),
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
            except Exception:
                pass
    _detector = fret_detector if fret_detector is not None else _make_default_fret_detector()
    try:
        det_result = _detector.detect(out["canon"], shear=out.get("shear"), hand_mask=out.get("hand_mask"))
    except Exception as exc:
        out["debug_info"]["fret_detector"] = _make_debug_info("fret_detector", exc, detector=type(_detector).__name__)
        fallback_detector = GeometricFretDetector() if not isinstance(_detector, GeometricFretDetector) else IntensityFretDetector(mode="max")
        try:
            det_result = fallback_detector.detect(out["canon"], shear=out.get("shear"), hand_mask=out.get("hand_mask"))
            out["debug_info"]["fret_detector_fallback"] = type(fallback_detector).__name__
        except Exception as exc2:
            out["fit"] = _make_empty_fit()
            out["invalid_reason"] = f"fret_detection_failed: {exc2}"
            out["debug_info"]["fret_detector_fallback_error"] = _make_debug_info("fret_detector_fallback", exc2)
            return out

    out["fret_xs_raw"]   = det_result.get("fret_xs_raw", [])
    out["fret_xs_filt"]  = det_result.get("fret_xs_filt", [])
    out["removed_pairs"] = det_result.get("removed_pairs", [])
    out["fit"]           = det_result.get("fit", _make_empty_fit())
    method_label = str(det_result.get("method", "unknown"))
    if method_label.startswith("intensity"):
        out["fret_detector_method"] = "intensity"
        out["fret_detector_detail"] = method_label
    elif method_label.startswith("geometric"):
        out["fret_detector_method"] = "geometric"
        out["fret_detector_detail"] = method_label
    else:
        out["fret_detector_method"] = method_label

    if "profile" in det_result:
        out["intensity_profile"] = det_result["profile"]
    if "profile_mode" in det_result:
        out["intensity_profile_mode"] = det_result["profile_mode"]
    if "auto_strategy" in det_result:
        out["intensity_auto_strategy"] = det_result["auto_strategy"]
    if "debug_info" in det_result:
        out["debug_info"]["fret_detector_detail"] = det_result["debug_info"]

    h_img, w_img = img.shape[:2]
    try:
        out["fingertips"] = step9_project_fingertips(
            landmarks, out["H"], w_img, h_img, fit=out.get("fit")
        )
    except Exception as exc:
        out["fingertips"] = []
        out["debug_info"]["fingertips"] = _make_debug_info("fingertips", exc)

    # Orientation normalization: detect if nut is on the right and flag for downstream use.
    # canon_norm is the canonical image guaranteed to have the nut on the left;
    # features.py mirrors x-coordinates when is_flipped=True.
    is_flipped = _derive_is_flipped(out.get("fit"), orientation, landmarks)
    out["is_flipped"] = is_flipped
    out["canon_norm"] = cv2.flip(out["canon"], 1) if is_flipped else out["canon"]
    out["nut_direction"] = "Nut-Right (flipped)" if is_flipped else "Nut-Left (standard)"

    out["ok"] = True
    return out
