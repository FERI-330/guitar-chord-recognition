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
    step1_canny, step2_hough, step3_neck_angle, step4_split_lines,
    step5_outer_edges, step6_clamp_trapezoid_extent, step6_trapezoid, step6_warp,
    step6b_find_nut, step6c_trim_to_nut,
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
    def detect(self, canon_bgr: np.ndarray, nut: Optional[dict] = None) -> dict:
        """Bundpozíciók detektálása a kanonikus képen.

        Args:
            canon_bgr: kanonikus perspektíva-warped BGR kép (600×80 px).
            nut:       nut detektálás eredménye (``step6b_find_nut`` kimenete),
                       vagy ``None`` ha nem elérhető.

        Returns:
            Dict kötelező kulcsokkal: fit, fret_xs_raw, fret_xs_filt,
            removed_pairs, method.
        """


class GeometricFretDetector(FretDetectorInterface):
    """Eredeti V14 Hough-alapú bunddetektálás (step7 + suppress + step8).

    Ez az osztály kizárólag átcsomagolja a meglévő ``geometry.py`` függvényeket –
    semmi logikát nem implementál újra.  Alapértelmezett detektor a pipeline-ban.
    """

    def detect(self, canon_bgr: np.ndarray, nut: Optional[dict] = None) -> dict:
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
    """Intenzitás-gradiens alapú bunddetektálás (Sobel-X csúcsdetektálás + step8).

    A Hough-transzformáció helyett a kanonikus kép oszloponkénti Sobel-X
    gradiens összegéből generál 1D profilt, majd ``scipy.signal.find_peaks``-kel
    detektál csúcsokat.  A 17.817-es illesztést ugyanúgy a meglévő
    ``step8_fit_fret_rule`` végzi – a matematikai logika nem kerül
    újraimplementálásra ebben az osztályban.

    Előnyök a geometriai módszerrel szemben:
    - Zajos / alacsony kontrasztú képeken jobban teljesíthet
    - Nem függ a Hough küszöb paramétereinek helyes beállításától
    - A gradiens-profil inspektálható (``result['profile']`` kulcs)

    Hátrányok:
    - Érzékenyebb ujj-okklúzióra (az ujjak is gradienst okoznak)
    - A ``suppress_pairs`` opció segít, de nem eliminálja teljesen
    """

    def __init__(
        self,
        sobel_ksize: int = 3,
        smooth_sigma: float = 1.5,
        peak_height: float = 0.12,
        peak_distance: int = 7,
        peak_prominence: float = 0.06,
        peak_max_width: float = 14.0,
        suppress_pairs: bool = True,
    ) -> None:
        """
        Args:
            sobel_ksize:      OpenCV Sobel kernel mérete (1, 3 vagy 5).
            smooth_sigma:     Gaussian simítás sigma értéke a gradiens-profilra.
            peak_height:      Minimális csúcsmagasság (normalizált 0–1 skálán).
            peak_distance:    Minimális csúcs–csúcs távolság (px).
            peak_prominence:  Minimális csúcs prominencia.
            peak_max_width:   Maximális csúcsszélesség (px); a szélesebb csúcsok
                              valószínűleg ujjak, nem bundok.
            suppress_pairs:   Ujjpár-szuppresszió alkalmazása (mint a geometriai
                              úton, ``suppress_finger_pairs``).
        """
        self.sobel_ksize    = sobel_ksize
        self.smooth_sigma   = smooth_sigma
        self.peak_height    = peak_height
        self.peak_distance  = peak_distance
        self.peak_prominence = peak_prominence
        self.peak_max_width = peak_max_width
        self.suppress_pairs = suppress_pairs

    def gradient_profile(self, canon_bgr: np.ndarray) -> np.ndarray:
        """Normalizált oszloponkénti Sobel-X gradiens összeg.

        Nyilvános metódus: vizualizálható a ``PipelineVisualizer``-ből.

        Returns:
            1D float array, hossza CANONICAL_W (600), értékek [0, 1].
        """
        from scipy.ndimage import gaussian_filter1d

        gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.sobel_ksize)
        profile = np.abs(sobel).sum(axis=0)
        if self.smooth_sigma > 0:
            profile = gaussian_filter1d(profile, sigma=self.smooth_sigma)
        max_val = float(profile.max())
        if max_val > 1e-3:
            profile = profile / max_val
        return profile.astype(np.float32)

    def detect(self, canon_bgr: np.ndarray, nut: Optional[dict] = None) -> dict:
        from scipy.signal import find_peaks

        profile = self.gradient_profile(canon_bgr)

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

        fit["method"] = "intensity"
        return {
            "fit":           fit,
            "fret_xs_raw":   fret_xs_raw,
            "fret_xs_filt":  fret_xs_filt,
            "removed_pairs": removed_pairs,
            "method":        "intensity",
            "profile":       profile,
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
                     fret_detector: Optional[FretDetectorInterface] = None) -> dict:
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

    # ── 6b. Trapézoid along-extent clamp a csuklóhoz ────────────────────────
    edge_info = step6_clamp_trapezoid_extent(edge_info, anchor)

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

    # ── 10. Nut detektálás (side_hint + hand_boundary alapú keresés) ─────────
    side_hint = _choose_nut_side(anchor, H, img.shape)
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

    # ── 11. Nut-trim + re-warp ──────────────────────────────────────────────
    if nut is not None:
        corners_trim = step6c_trim_to_nut(trap["corners_px"], H_inv, nut)
        H2, H2_inv, canon2 = step6_warp(img, corners_trim)
        out["corners_trim"] = corners_trim
        out["H"], out["H_inv"], out["canon"] = H2, H2_inv, canon2

    # ── 12–14. Bunddetektálás (cserélhető detektor) ─────────────────────────
    _detector = fret_detector if fret_detector is not None else _make_default_fret_detector()
    try:
        det_result = _detector.detect(out["canon"], nut=out.get("nut"))
        out["fret_xs_raw"]          = det_result["fret_xs_raw"]
        out["fret_xs_filt"]         = det_result["fret_xs_filt"]
        out["removed_pairs"]        = det_result["removed_pairs"]
        out["fit"]                  = det_result["fit"]
        out["fret_detector_method"] = det_result.get("method", "unknown")
        # IntensityFretDetector esetén a gradiens-profil is elérhető
        if "profile" in det_result:
            out["intensity_profile"] = det_result["profile"]
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
