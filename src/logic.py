from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import CFG


FINGER_TIP_INDEX = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

FINGER_LABEL_HU = {
    "thumb": "Hüvelykujj",
    "index": "Mutatóujj",
    "middle": "Középső",
    "ring": "Gyűrűs",
    "pinky": "Kisujj",
}

_DEFAULT_TOUCH_POINT_OFFSET_RATIO = float(CFG.get("touch_point_offset_ratio", 0.025))


def _resolve_touch_point_offset_px(
    touch_point_offset_ratio: Optional[float],
    touch_point_offset_px: Optional[float],
    canon_height: Optional[float],
) -> float:
    """Resolve the vertical touch-point offset in canonical pixels."""
    if touch_point_offset_px is not None:
        try:
            return max(0.0, float(touch_point_offset_px))
        except Exception:
            return 0.0

    ratio = _DEFAULT_TOUCH_POINT_OFFSET_RATIO if touch_point_offset_ratio is None else touch_point_offset_ratio
    try:
        ratio_value = max(0.0, float(ratio))
    except Exception:
        ratio_value = _DEFAULT_TOUCH_POINT_OFFSET_RATIO

    if canon_height is None:
        return 0.0

    return max(0.0, float(canon_height) * ratio_value)


def _as_point_array(mp_results: Any, img_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Convert MediaPipe landmarks into an Nx2 array.

    Supports:
    - MediaPipe hand landmarks objects with `.landmark`
    - a flat iterable of landmark-like objects with `.x` / `.y`
    - dicts with `x`, `y`
    - `(x, y)` tuples/lists
    - pre-projected Nx2 arrays
    """
    if mp_results is None:
        return None

    points = None

    if hasattr(mp_results, "landmark"):
        points = list(mp_results.landmark)
    elif isinstance(mp_results, np.ndarray):
        if mp_results.ndim == 2 and mp_results.shape[1] >= 2:
            points = mp_results[:, :2].astype(np.float32)
    elif isinstance(mp_results, Sequence) and not isinstance(mp_results, (str, bytes)):
        points = list(mp_results)

    if points is None:
        return None

    coords: List[Tuple[float, float]] = []
    for point in points:
        if point is None:
            coords.append((math.nan, math.nan))
            continue
        if hasattr(point, "x") and hasattr(point, "y"):
            coords.append((float(point.x), float(point.y)))
        elif isinstance(point, Mapping) and "x" in point and "y" in point:
            coords.append((float(point["x"]), float(point["y"])))
        elif isinstance(point, (tuple, list, np.ndarray)) and len(point) >= 2:
            coords.append((float(point[0]), float(point[1])))
        else:
            coords.append((math.nan, math.nan))

    arr = np.asarray(coords, dtype=np.float32)

    if img_shape is not None and arr.size and np.nanmax(arr) <= 1.01:
        height, width = img_shape[:2]
        arr[:, 0] *= float(width)
        arr[:, 1] *= float(height)

    return arr


def _to_canonical(points_px: np.ndarray, detection_results: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Project points into canonical ROI coordinates using the detection homography."""
    if points_px is None or len(points_px) == 0:
        return None

    H_inv = detection_results.get("H_inv")
    H = detection_results.get("H")

    if H_inv is None and H is None:
        return points_px

    if H_inv is None and H is not None:
        try:
            H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
        except Exception:
            return None

    try:
        pts = points_px.reshape(-1, 1, 2).astype(np.float32)
        projected = cv2.perspectiveTransform(pts, np.asarray(H_inv, dtype=np.float64))
        return projected.reshape(-1, 2)
    except Exception:
        return None


def _normalize_fret_xs(detection_results: Mapping[str, Any]) -> List[float]:
    """Collect and sort fret x-positions from detection results."""
    fit = detection_results.get("fit") or {}
    predicted_x = fit.get("predicted_x")

    fret_xs: Iterable[Any]
    if isinstance(predicted_x, Mapping):
        fret_xs = predicted_x.values()
    elif isinstance(predicted_x, (list, tuple, np.ndarray)):
        fret_xs = predicted_x
    else:
        fret_xs = detection_results.get("fret_xs_filt", []) or []

    out: List[float] = []
    for value in fret_xs:
        try:
            out.append(float(value))
        except Exception:
            continue
    out.sort()
    return out


def _finger_fret_from_x(x: float, nut_x: Optional[float], fret_xs: Sequence[float]) -> Any:
    """Assign a fret number based on canonical x position."""
    if nut_x is not None and x < nut_x:
        return "OUT"
    if not fret_xs:
        return "OUT"

    for fret_idx, fret_x in enumerate(fret_xs, start=1):
        if x < fret_x:
            return fret_idx

    return "OUT"


def map_fingers_to_frets(mp_results: Any, detection_results: Mapping[str, Any]) -> Dict[str, Any]:
    """Map MediaPipe fingertip positions to the detected fret intervals.

    Returns a dict with keys `thumb`, `index`, `middle`, `ring`, `pinky`.
    Each value contains:
    - `fret`: integer fret number or `OUT`
    - `canon_xy`: projected touch-point position in canonical ROI coordinates
    - `tip_xy`: raw projected fingertip position before offset correction
    - `touch_xy`: offset-corrected touch-point position used for fret lookup
    - `label`: Hungarian finger label
    """
    img = detection_results.get("img")
    canon = detection_results.get("canon")
    canon_shape = canon.shape if getattr(canon, "shape", None) is not None else None
    img_shape = getattr(img, "shape", None)

    points_px = _as_point_array(mp_results, img_shape=img_shape)
    if points_px is None:
        fingertips = detection_results.get("fingertips")
        if fingertips is not None:
            points_px = _as_point_array(fingertips, img_shape=canon_shape)

    if points_px is None:
        return {
            name: {"fret": "OUT", "canon_xy": None, "tip_xy": None, "touch_xy": None, "label": FINGER_LABEL_HU[name]}
            for name in FINGER_TIP_INDEX
        }

    touch_point_offset_px = _resolve_touch_point_offset_px(
        detection_results.get("touch_point_offset_ratio"),
        detection_results.get("touch_point_offset_px"),
        float(img_shape[0]) if img_shape is not None else (float(canon_shape[0]) if canon_shape is not None else None),
    )

    touch_points_px = np.array(points_px, dtype=np.float32, copy=True)
    for tip_idx in FINGER_TIP_INDEX.values():
        if 0 <= tip_idx < len(touch_points_px):
            touch_points_px[tip_idx, 1] += touch_point_offset_px

    tip_canon_points = _to_canonical(points_px, detection_results)
    if tip_canon_points is None:
        tip_canon_points = points_px

    canon_points = _to_canonical(touch_points_px, detection_results)
    if canon_points is None:
        canon_points = touch_points_px

    fret_xs = _normalize_fret_xs(detection_results)

    nut = detection_results.get("nut") or {}
    nut_x = None
    if isinstance(nut, Mapping):
        try:
            nut_x = float(nut.get("nut_x"))
        except Exception:
            nut_x = None

    if canon_shape is not None:
        height, width = canon_shape[:2]
    else:
        height = width = None

    result: Dict[str, Any] = {}
    for finger_name, landmark_idx in FINGER_TIP_INDEX.items():
        if landmark_idx >= len(canon_points):
            result[finger_name] = {
                "fret": "OUT",
                "canon_xy": None,
                "tip_xy": None,
                "touch_xy": None,
                "touch_offset_px": touch_point_offset_px,
                "label": FINGER_LABEL_HU[finger_name],
            }
            continue

        tip_x = float(tip_canon_points[landmark_idx, 0])
        tip_y = float(tip_canon_points[landmark_idx, 1])
        touch_x = tip_x
        touch_y = tip_y + touch_point_offset_px

        if height is not None:
            touch_y = min(max(0.0, touch_y), float(height) - 1.0)
        if width is not None:
            touch_x = min(max(0.0, touch_x), float(width) - 1.0)

        tip_xy = (tip_x, tip_y)
        touch_xy = (touch_x, touch_y)

        if math.isnan(tip_x) or math.isnan(tip_y):
            fret_value: Any = "OUT"
        elif width is not None and height is not None and not (0.0 <= touch_x < width and 0.0 <= touch_y < height):
            fret_value = "OUT"
        else:
            fret_value = _finger_fret_from_x(touch_x, nut_x, fret_xs)

        result[finger_name] = {
            "fret": fret_value,
            "canon_xy": touch_xy,
            "tip_xy": tip_xy,
            "touch_xy": touch_xy,
            "touch_offset_px": touch_point_offset_px,
            "label": FINGER_LABEL_HU[finger_name],
        }

    result["_debug"] = {
        "fret_xs": fret_xs,
        "nut_x": nut_x,
        "canon_shape": canon_shape,
        "touch_point_offset_px": touch_point_offset_px,
        "touch_point_offset_ratio": detection_results.get("touch_point_offset_ratio", _DEFAULT_TOUCH_POINT_OFFSET_RATIO),
    }
    return result
