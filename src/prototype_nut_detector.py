"""
src/prototype_nut_detector.py

Opcionális nut (gitár nyereg / 0. bund) detektor — CSAK vizualizációhoz.

FONTOS: Ennek a modulnak az eredménye SOHA nem áramolhat be a FretDetectorba
vagy az ML feature vektorokba. Kizárólag diagnosztikai / debug megjelenítésre
szolgál (szaggatott sárga vonal a kanonikus képen).

A nut detekció ki lett vezetve a kritikus pipeline-útból (F2 fázis refaktor).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.config import CFG
from src.geometry import step6b_find_nut
from src.hand_landmark import get_fretboard_near_edge


def _make_safety_nut(canon_bgr: np.ndarray,
                     side_hint: Optional[str] = None,
                     margin_px: int = 4) -> dict:
    """Fallback nut anchor a ROI széle közelében, ha step6b_find_nut nem talál semmit."""
    width = int(canon_bgr.shape[1]) if canon_bgr is not None else int(CFG["canonical_w"])
    side = side_hint if side_hint in ("left", "right") else "left"
    nut_x = margin_px if side == "left" else max(0, width - 1 - margin_px)
    return {
        "side": side,
        "nut_x": float(nut_x),
        "peak": 0.0,
        "ratio": 0.0,
        "width_px": 0.0,
        "col_response": None,
        "safety": True,
    }


def _project_landmark_to_canon(lm_px: tuple[float, float],
                                H: np.ndarray) -> Optional[float]:
    """Pixel-koordinátát H homográfián vetít a kanonikus térbe. Visszaad: canon x vagy None."""
    pt = np.array([lm_px[0], lm_px[1], 1.0])
    proj = H @ pt
    if abs(proj[2]) < 1e-9:
        return None
    return float(proj[0] / proj[2])


def detect_nut_prototype(result: dict) -> Optional[dict]:
    """Nut pozíció becslése egy kész pipeline result dict alapján — CSAK vizualizációhoz.

    A visszaadott dict SOHA nem kerülhet be a FretDetector-ba vagy feature vektorokba.

    Args:
        result: run_v14_pipeline() visszatérési értéke.

    Returns:
        Nut dict (``step6b_find_nut`` formátumú), vagy None ha nem detektálható.
        Kulcsok: side, nut_x, peak, ratio, width_px, safety.
    """
    canon = result.get("canon")
    if canon is None:
        return None

    H = result.get("H")
    landmarks = result.get("landmarks")
    side_hint = result.get("nut_side_hint")

    # Kéz gitárnyak-felőli határának kanonikus x-pozíciója (opcionális hint)
    hand_bnd_x: Optional[float] = None
    if landmarks is not None and H is not None:
        try:
            img = result.get("img")
            img_shape = img.shape if img is not None else (480, 640)
            near_edge = get_fretboard_near_edge(landmarks, img_shape)
            if near_edge is not None:
                hand_bnd_x = _project_landmark_to_canon(near_edge, H)
        except Exception:
            pass

    try:
        nut = step6b_find_nut(canon, side_hint=side_hint, hand_boundary_canon_x=hand_bnd_x)
    except Exception:
        nut = None

    if nut is None:
        nut = _make_safety_nut(canon, side_hint=side_hint)
        nut["reason"] = "fallback_safety_nut"

    return nut


def detect_inlays_prototype(result: dict) -> list:
    """Kísérleti inlay-detektálás a kanonikus képen — CSAK vizualizációhoz.

    Keresi a 'dupla kis csúcs' mintázatot: 5–15 px szélességű, alacsony
    amplitúdójú Sobel-X csúcspárokat, amelyek gitár nyakjelző (inlay) pontok
    két szélét reprezentálhatják.

    A `hand_mask` (kanonikus tér, uint8) oszlopait lenullázza a profil-elemzés
    előtt, hogy az ujjak ne generálhassanak hamis inlay-jelölteket.

    Args:
        result: run_v14_pipeline() visszatérési értéke.

    Returns:
        list[dict] — minden dict = {canon_x, pair, confidence, heights}.
        Üres lista ha canon nem elérhető, vagy nem találhatók párok.
    """
    canon = result.get("canon")
    if canon is None:
        return []
    try:
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        col_profile = np.abs(sx).sum(axis=0)
        mx = float(col_profile.max())
        if mx < 1e-3:
            return []
        col_profile = col_profile / mx

        # Ujj-maszk alapú elnyomás: ahol a kézmaszk > 0, ott a profil 0-ra áll.
        # Ugyanolyan logikával, mint az IntensityFretDetector.detect()-ben.
        hand_mask = result.get("hand_mask")
        if hand_mask is not None:
            col_has_hand = np.any(hand_mask > 0, axis=0)
            if col_has_hand.shape[0] == col_profile.shape[0]:
                col_profile[col_has_hand] = 0.0

        col_profile = gaussian_filter1d(col_profile, sigma=0.8)

        peaks, _ = find_peaks(
            col_profile,
            height=0.04,
            distance=3,
            prominence=0.015,
            width=(1.0, 8.0),
        )

        inlays = []
        used: set = set()
        for i, p1 in enumerate(peaks):
            if i in used:
                continue
            for j in range(i + 1, len(peaks)):
                if j in used:
                    continue
                p2 = int(peaks[j])
                sep = float(p2 - p1)
                if sep > 15.0:
                    break
                if sep >= 5.0:
                    h1 = float(col_profile[p1])
                    h2 = float(col_profile[p2])
                    if h1 < 0.55 and h2 < 0.55:
                        inlays.append({
                            "canon_x": float((p1 + p2) / 2.0),
                            "pair": (float(p1), float(p2)),
                            "confidence": float(min(h1, h2) / (max(h1, h2) + 1e-9)),
                            "heights": (h1, h2),
                        })
                        used.add(i)
                        used.add(j)
                        break
        return inlays
    except Exception:
        return []
