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
