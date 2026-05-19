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
    step6b_find_nut, step6c_trim_to_nut, step6_extend_for_nut,
    step6d_shear_correction,
    step7_fret_lines_canonical, step8_fit_fret_rule,
)
from src.hand_landmark import (
    get_landmarker,
    step9_detect_landmarks, step9_project_fingertips,
    build_finger_mask, anchor_neck_angle, step3_neck_angle_anchored,
    get_fretboard_near_edge,
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
               shear: Optional[dict] = None) -> dict:
        """Bundpozíciók detektálása a kanonikus képen.

        Args:
            canon_bgr: kanonikus perspektíva-warped BGR kép (600×80 px).
            nut:       nut detektálás eredménye (``step6b_find_nut`` kimenete),
                       vagy ``None`` ha nem elérhető.
            shear:     step6d_shear_correction kimenete – az auto-mode dönti el
                       Sobel-X vs Max-pooling választást.

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
               shear: Optional[dict] = None) -> dict:
        fret_xs_raw = step7_fret_lines_canonical(canon_bgr)
        fret_xs_filt, removed_pairs = suppress_finger_pairs(fret_xs_raw)

        nut_side = nut["side"] if nut else None
        refine_enabled = bool(CFG.get("fret_refine_enabled", True))
        refine_tol = float(CFG.get("fret_refine_tol_px", 12.0))

        try:
            fit_pass1 = step8_fit_fret_rule(
                fret_xs_filt,
                nut_anchored=(nut_side is not None),
                nut_side=nut_side,
            )
            if refine_enabled:
                fret_xs_filt = refine_frets_by_fit(fret_xs_filt, fit_pass1, refine_tol)
                fit = step8_fit_fret_rule(
                    fret_xs_filt,
                    nut_anchored=(nut_side is not None),
                    nut_side=nut_side,
                )
            else:
                fit = fit_pass1
        except Exception as exc:
            print(f"  [GeometricFretDetector] step8 hiba: {exc}")
            fit = _make_empty_fit()

        fit["method"] = "geometric"
        return {
            "fit":           fit,
            "fret_xs_raw":   fret_xs_raw,
            "fret_xs_filt":  fret_xs_filt,
            "removed_pairs": removed_pairs,
            "method":        "geometric",
        }


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
    ) -> None:
        self.mode            = mode
        self.sobel_ksize     = sobel_ksize
        self.smooth_sigma    = smooth_sigma
        self.peak_height     = peak_height
        self.peak_distance   = peak_distance
        self.peak_prominence = peak_prominence
        self.peak_max_width  = peak_max_width
        self.suppress_pairs  = suppress_pairs

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

    def _snr(self, profile: np.ndarray, thr: float = 0.30) -> float:
        above = profile[profile >= thr]
        below = profile[profile <  thr]
        if len(above) == 0 or len(below) == 0:
            return 0.0
        return float(above.mean() / (below.mean() + 1e-9))

    def _select_mode(self, shear: Optional[dict]) -> str:
        """Shear-eredmény alapján Sobel-X vagy Max-pooling választása.

        Sobel-X (precizitás): shear korrigált VAGY bizonyítottan ≈0 (n≥4, |α|<0.2°)
        Max-pooling (robusztus): ismeretlen/maradt dőlés (kevés vonal, stb.)
        """
        if shear is None:
            return "sobel"
        corrected = bool(shear.get("corrected", False))
        angle     = abs(float(shear.get("shear_angle_deg", 0.0)))
        n_lines   = int(shear.get("n_lines", 0))
        if corrected or (n_lines >= 4 and angle < 0.2):
            return "sobel"
        return "max"

    # ── Nyilvános API ─────────────────────────────────────────────────────────

    def gradient_profile(self, canon_bgr: np.ndarray,
                         shear: Optional[dict] = None) -> np.ndarray:
        """Normalizált 1D profil a konfigurált módban.

        Nyilvános metódus: vizualizálható a PipelineVisualizer-ből.
        """
        gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        active = self.mode if self.mode != "auto" else self._select_mode(shear)
        if active == "max":
            return self._max_profile(gray)
        return self._sobel_profile(gray)

    def detect(self, canon_bgr: np.ndarray,
               nut: Optional[dict] = None,
               shear: Optional[dict] = None) -> dict:
        from scipy.signal import find_peaks

        gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        active_mode = self.mode if self.mode != "auto" else self._select_mode(shear)

        if active_mode == "max":
            profile = self._max_profile(gray)
        else:
            profile = self._sobel_profile(gray)

        # SNR-alapú fallback: ha Sobel profil gyenge, Max-pooling-ra vált
        if active_mode == "sobel" and self.mode == "auto":
            snr_val = self._snr(profile)
            if snr_val < self._SNR_FALLBACK_THR:
                profile = self._max_profile(gray)
                active_mode = "max"
                print(f"  [IntensityFretDetector] Sobel SNR={snr_val:.2f} → Max-pooling fallback")

        _shear_info = "n/a" if shear is None else f"{shear.get('shear_angle_deg', 0):.2f}°"
        print(f"  [IntensityFretDetector] mode={active_mode} | shear={_shear_info}")

        peak_idxs, _ = find_peaks(
            profile,
            height=self.peak_height,
            distance=self.peak_distance,
            prominence=self.peak_prominence,
            width=(0.0, self.peak_max_width),
        )

        fret_xs_raw = [float(i) for i in peak_idxs]

        if self.suppress_pairs:
            fret_xs_filt, removed_pairs = suppress_finger_pairs(fret_xs_raw)
        else:
            fret_xs_filt, removed_pairs = fret_xs_raw, []

        nut_side = nut["side"] if nut else None
        refine_enabled = bool(CFG.get("fret_refine_enabled", True))
        refine_tol = float(CFG.get("fret_refine_tol_px", 12.0))

        try:
            fit_pass1 = step8_fit_fret_rule(
                fret_xs_filt,
                nut_anchored=(nut_side is not None),
                nut_side=nut_side,
            )
            if refine_enabled:
                fret_xs_filt = refine_frets_by_fit(fret_xs_filt, fit_pass1, refine_tol)
                fit = step8_fit_fret_rule(
                    fret_xs_filt,
                    nut_anchored=(nut_side is not None),
                    nut_side=nut_side,
                )
            else:
                fit = fit_pass1
        except Exception as exc:
            print(f"  [IntensityFretDetector] step8 hiba: {exc}")
            fit = _make_empty_fit()

        fit["method"] = f"intensity_{active_mode}"
        return {
            "fit":           fit,
            "fret_xs_raw":   fret_xs_raw,
            "fret_xs_filt":  fret_xs_filt,
            "removed_pairs": removed_pairs,
            "method":        f"intensity_{active_mode}",
            "profile":       profile,
            "profile_mode":  active_mode,
        }


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
    }

    # ── Kép betöltés ────────────────────────────────────────────────────────
    try:
        img = load_image_bgr(img_entry["path"])
    except FileNotFoundError as e:
        out["invalid_reason"] = "load_failed"
        return out
    out["img"] = img

    # ── 1. MediaPipe landmarks ───────────────────────────────────────────────
    # (Landmarks detektálása az eredeti képen fut — CLAHE ronthatja a MediaPipe teljesítményét)
    landmarks = step9_detect_landmarks(img_entry["path"], landmarker)
    out["landmarks"] = landmarks

    # ── 2. Anchor + ujjmaszk ────────────────────────────────────────────────
    anchor = anchor_neck_angle(landmarks, img.shape)
    out["anchor"] = anchor
    finger_mask = build_finger_mask(img.shape, landmarks)
    out["finger_mask"] = finger_mask

    # ── Pre-pipeline előfeldolgozás (opcionális) ─────────────────────────────
    if preprocessor is not None:
        img = preprocessor.process(img)
        out["img_preprocessed"] = img

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

    # Nyakirány finomítás bund-vonalakból (step3b).
    # Megjegyzés: az alapértelmezett hough_min_len_frac=0.15 általában szűri a rövid
    # fret-vonalakat, így ez a blokk csak kisebb frac érték esetén aktív.
    if len(split["fret_lines"]) >= 3:
        refined_angle = step3b_refine_neck_angle(neck["angle_deg"], split["fret_lines"])
        if abs(refined_angle - neck["angle_deg"]) > 0.1:
            split = step4_split_lines(lines, refined_angle)
            neck = dict(neck)
            neck["angle_deg"] = refined_angle
            neck["angle_refined"] = True

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

    # ── 6b. Trapézoid along-extent clamp a csuklóhoz ────────────────────────
    edge_info = step6_clamp_trapezoid_extent(edge_info, anchor)

    # ── 7. Trapézoid (Nut-First: landmarks alapján kiterjesztett a_min) ────────
    trap = step6_trapezoid(img, edge_info, landmarks=landmarks)
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

    # ── 10. Nut detektálás (side_hint + hand_boundary alapú keresés) ─────────
    side_hint = _choose_nut_side(anchor, H, img.shape, landmarks=landmarks)
    out["nut_side_hint"] = side_hint

    # Kézél vetítése a kanonikus térbe → Nut-keresési sáv korlátozása
    hand_bnd_x: Optional[float] = None
    near_edge = get_fretboard_near_edge(landmarks, img.shape)
    if near_edge is not None:
        hand_bnd_x = _project_landmark_to_canon(near_edge, H)
    out["hand_boundary_canon_x"] = hand_bnd_x

    nut = step6b_find_nut(canon, side_hint=side_hint,
                          hand_boundary_canon_x=hand_bnd_x)
    out["nut"] = nut

    # ── 10b. Nut fallback: ha nem detektálható, ROI bővítés és újra-keresés ─
    if nut is None and side_hint is not None:
        extend_px = int(CFG.get("nut_fallback_extend_px", 80))
        corners_ext = step6_extend_for_nut(trap["corners_px"], H_inv, side_hint, extend_px)
        if corners_ext is not None:
            H_ext, H_ext_inv, canon_ext = step6_warp(img, corners_ext)
            nut_ext = step6b_find_nut(canon_ext, side_hint=side_hint,
                                      hand_boundary_canon_x=hand_bnd_x)
            if nut_ext is not None:
                H, H_inv, canon = H_ext, H_ext_inv, canon_ext
                out["H"], out["H_inv"], out["canon"] = H, H_inv, canon
                out["nut"] = nut_ext
                nut = nut_ext
                print(f"  [nut_fallback] nut találat kiterjesztett ROI-ban @ x={nut['nut_x']}px")

    # ── 11. Nut-trim + re-warp ──────────────────────────────────────────────
    if nut is not None:
        corners_trim = step6c_trim_to_nut(trap["corners_px"], H_inv, nut)
        H2, H2_inv, canon2 = step6_warp(img, corners_trim)
        out["corners_trim"] = corners_trim
        out["H"], out["H_inv"], out["canon"] = H2, H2_inv, canon2

    # ── 11b. Post-warp shear korrekció (step6d) ────────────────────────────
    out["canon_pre_shear"] = out["canon"]   # fallback-összehasonlításhoz
    shear = step6d_shear_correction(out["canon"])
    out["shear"] = shear
    if shear["corrected"]:
        out["H"]     = shear["S"] @ out["H"]
        out["H_inv"] = out["H_inv"] @ shear["S_inv"]
        out["canon"] = shear["canon_corrected"]
        # Nut újra-detektálása a shear-korrigált kanonikus képen.
        # A shear-korrekció a nut x-pozícióját eltolhatja (x_dst = -s·y),
        # ezért a pre-shear nut_x nem használható a step8 anchor-ként.
        nut_post = step6b_find_nut(
            out["canon"],
            side_hint=out.get("nut_side_hint"),
            hand_boundary_canon_x=out.get("hand_boundary_canon_x"),
        )
        if nut_post is not None:
            out["nut"] = nut_post
            nut = nut_post

    # ── 12–14. Bunddetektálás (cserélhető detektor) ─────────────────────────
    _detector = fret_detector if fret_detector is not None else _make_default_fret_detector()
    try:
        det_result = _detector.detect(
            out["canon"], nut=out.get("nut"), shear=out.get("shear")
        )
        out["fret_xs_raw"]          = det_result["fret_xs_raw"]
        out["fret_xs_filt"]         = det_result["fret_xs_filt"]
        out["removed_pairs"]        = det_result["removed_pairs"]
        out["fit"]                  = det_result["fit"]
        out["fret_detector_method"] = det_result.get("method", "unknown")
        # IntensityFretDetector esetén a gradiens-profil és aktív mód is elérhető
        if "profile" in det_result:
            out["intensity_profile"] = det_result["profile"]
        if "profile_mode" in det_result:
            out["intensity_profile_mode"] = det_result["profile_mode"]
    except Exception as exc:
        out["fit"] = None
        out["invalid_reason"] = f"fret_detection_failed: {exc}"
        return out

    # ── 15. Ujjhegy vetítés (ha van landmark és H) ──────────────────────────
    h_img, w_img = img.shape[:2]
    out["fingertips"] = step9_project_fingertips(
        landmarks, out["H"], w_img, h_img, fit=out.get("fit")
    )

    out["ok"] = True
    return out
