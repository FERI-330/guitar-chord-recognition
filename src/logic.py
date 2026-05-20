from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


FINGER_TIP_INDEX = {
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

FINGER_LABEL_HU = {
    "index": "Mutatóujj",
    "middle": "Középső",
    "ring": "Gyűrűs",
    "pinky": "Kisujj",
}


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

    Returns a dict with keys `index`, `middle`, `ring`, `pinky`.
    Each value contains:
    - `fret`: integer fret number or `OUT`
    - `canon_xy`: projected fingertip position in canonical ROI coordinates
    - `label`: Hungarian finger label
    """
    img = detection_results.get("img")
    canon = detection_results.get("canon")
    canon_shape = canon.shape if getattr(canon, "shape", None) is not None else None

    points_px = _as_point_array(mp_results, img_shape=getattr(img, "shape", None))
    if points_px is None:
        fingertips = detection_results.get("fingertips")
        if fingertips is not None:
            points_px = _as_point_array(fingertips, img_shape=canon_shape)

    if points_px is None:
        return {
            name: {"fret": "OUT", "canon_xy": None, "label": FINGER_LABEL_HU[name]}
            for name in FINGER_TIP_INDEX
        }

    canon_points = _to_canonical(points_px, detection_results)
    if canon_points is None:
        canon_points = points_px

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
                "label": FINGER_LABEL_HU[finger_name],
            }
            continue

        x = float(canon_points[landmark_idx, 0])
        y = float(canon_points[landmark_idx, 1])
        canon_xy = (x, y)

        if math.isnan(x) or math.isnan(y):
            fret_value: Any = "OUT"
        elif width is not None and height is not None and not (0.0 <= x < width and 0.0 <= y < height):
            fret_value = "OUT"
        else:
            fret_value = _finger_fret_from_x(x, nut_x, fret_xs)

        result[finger_name] = {
            "fret": fret_value,
            "canon_xy": canon_xy,
            "label": FINGER_LABEL_HU[finger_name],
        }

    result["_debug"] = {
        "fret_xs": fret_xs,
        "nut_x": nut_x,
        "canon_shape": canon_shape,
    }
    return result
