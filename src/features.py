"""
src/features.py

Feature vektor összeállítása a V14 pipeline kimenetéből, batch extrakció, NPZ mentés/betöltés.

Feature vektor (56 dim):
  Group B (42): wrist-normalized landmark x,y  ← LEGFONTOSABB, ok=False esetén is megmarad
  Group D ( 2): detection flags (hand_detected, fretboard_detected)
  Group F ( 2): neck angle cos/sin
  Group G ( 5): ujj-bund index normalizálva (0=nem detektált)
  Group H ( 5): ujj-húr pozíció (0=nem detektált)

Failed detection policy:
  ok=False → Group B megmarad (ha landmarks nem None), G/H=0, D=(hand,0), F=0
  landmarks=None → minden 0
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import CFG, PATHS
from src.constants import CANONICAL_H, N_FRETS, FINGER_TIP_IDX

# ── Feature csoport méretek ────────────────────────────────────────────────
GROUP_B_SIZE = 42   # wrist-centered x,y (21 landmark × 2)
GROUP_D_SIZE = 2    # flags: hand_detected, fretboard_detected
GROUP_F_SIZE = 2    # neck angle: cos, sin
GROUP_G_SIZE = 5    # fret index per finger (normalizált 0–1)
GROUP_H_SIZE = 5    # string norm per finger (0–1)

FEATURE_DIM = GROUP_B_SIZE + GROUP_D_SIZE + GROUP_F_SIZE + GROUP_G_SIZE + GROUP_H_SIZE  # 56

# Fret quality threshold: ha a fit coverage_ratio alatta van → fretboard_detected=0
COVERAGE_THRESHOLD = 0.40

# Offset-ek a feature vektorban
_OFF_B = 0
_OFF_D = _OFF_B + GROUP_B_SIZE          # 42
_OFF_F = _OFF_D + GROUP_D_SIZE          # 44
_OFF_G = _OFF_F + GROUP_F_SIZE          # 46
_OFF_H = _OFF_G + GROUP_G_SIZE          # 51
# végpont: 56


def _group_b(landmarks: list) -> np.ndarray:
    """Wrist-normalized landmark x,y (42 dim).

    A csukló (lm[0]) körüli kéz-centroid koordináták, hand_scale-re normalizálva.
    hand_scale = euklidészi távolság csukló–középső MCP (lm[0]–lm[9]).
    """
    vec = np.zeros(GROUP_B_SIZE, dtype=np.float32)
    if not landmarks or len(landmarks) < 21:
        return vec
    pts = np.array([[lx, ly] for (lx, ly, _) in landmarks], dtype=np.float32)
    wrist = pts[0]
    hand_scale = float(np.linalg.norm(pts[9] - wrist))
    if hand_scale < 1e-6:
        hand_scale = 1.0
    centered = (pts - wrist) / hand_scale
    vec[:] = centered.flatten()
    return vec


def _group_f(neck: Optional[dict]) -> np.ndarray:
    """Neck angle cos/sin (2 dim)."""
    vec = np.zeros(GROUP_F_SIZE, dtype=np.float32)
    if neck is None:
        return vec
    angle_rad = np.radians(neck.get("angle_deg", 0.0))
    vec[0] = float(np.cos(angle_rad))
    vec[1] = float(np.sin(angle_rad))
    return vec


def _group_gh(fingertips: list, ok: bool) -> tuple[np.ndarray, np.ndarray]:
    """Fret index (G, 5 dim) és string norm (H, 5 dim) per ujj."""
    g = np.zeros(GROUP_G_SIZE, dtype=np.float32)
    h = np.zeros(GROUP_H_SIZE, dtype=np.float32)
    if not ok or not fingertips:
        return g, h
    tip_map = {fp["tip_idx"]: fp for fp in fingertips}
    for col_i, tip_idx in enumerate(FINGER_TIP_IDX):  # [4,8,12,16,20]
        if tip_idx not in tip_map:
            continue
        fp = tip_map[tip_idx]
        fret_est = fp.get("fret_est")
        if fret_est is not None:
            g[col_i] = float(np.clip(fret_est / N_FRETS, 0.0, 1.0))
        h[col_i] = float(np.clip(fp.get("string_norm", 0.0), 0.0, 1.0))
    return g, h


def assemble_feature_vector(result: dict) -> np.ndarray:
    """V14 pipeline result dict → 56-dimenziós feature vektor.

    A teljes pipeline kimenetét (run_v14_pipeline visszatérési értékét) vár.
    Mindig 56-dimenziós vektort ad vissza, még ok=False esetén is.
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    landmarks = result.get("landmarks")
    ok = bool(result.get("ok", False))
    neck = result.get("neck") if ok else None
    fingertips = result.get("fingertips", []) if ok else []
    fit = result.get("fit") or {}
    coverage = float(fit.get("coverage_ratio", 0.0))
    is_flipped = bool(result.get("is_flipped", False))

    hand_detected = 1.0 if landmarks is not None else 0.0
    fretboard_detected = 1.0 if (ok and coverage >= COVERAGE_THRESHOLD) else 0.0

    # Group B – wrist-normalized landmarks; negate x when flipped so the
    # classifier always sees a standard nut-left hand pose.
    b = _group_b(landmarks)
    if is_flipped:
        b[0::2] = -b[0::2]   # x components are at even indices
    vec[_OFF_B:_OFF_D] = b

    # Group D – flags
    vec[_OFF_D] = hand_detected
    vec[_OFF_D + 1] = fretboard_detected

    # Group F – neck angle; horizontal flip negates the sin component (slope sign).
    f = _group_f(neck)
    if is_flipped:
        f[1] = -f[1]
    vec[_OFF_F:_OFF_G] = f

    # Group G + H – fret/string (csak ok=True AND jó coverage esetén)
    # fret_est is orientation-agnostic (fret number), string_norm is y-based → no flip needed.
    if fretboard_detected > 0.5:
        g, h = _group_gh(fingertips, ok)
        vec[_OFF_G:_OFF_H] = g
        vec[_OFF_H:] = h

    return vec


def feature_names() -> list[str]:
    """A 56-dimenziós feature vektor dimenzióneveinek listája."""
    lm_names = ["thumb", "index_mcp", "index_pip", "index_dip", "index_tip",
                 "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
                 "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
                 "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
                 "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip"]
    # MediaPipe: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    lm_order = ["wrist",
                 "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                 "index_mcp", "index_pip", "index_dip", "index_tip",
                 "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
                 "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
                 "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
    names = []
    for lm in lm_order:
        names += [f"B_{lm}_x", f"B_{lm}_y"]
    names += ["D_hand_detected", "D_fretboard_detected"]
    names += ["F_neck_cos", "F_neck_sin"]
    finger_labels = ["thumb", "index", "middle", "ring", "pinky"]
    names += [f"G_fret_{f}" for f in finger_labels]
    names += [f"H_str_{f}" for f in finger_labels]
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Batch extrakció
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_batch(manifest_df: pd.DataFrame,
                           landmarker=None,
                           verbose: bool = True) -> dict:
    """Lefuttatja a pipeline-t minden képre és összeállítja a feature mátrixot.

    Visszaad:
        {
          'X':       (N, 56) float32 feature mátrix,
          'y':       (N,)    int osztálycímkék (0..7),
          'classes': list[str]  osztálynevek (y → classes[y]),
          'paths':   list[str]  képútvonalak,
          'splits':  list[str]  'train'/'val'/'test',
          'ok_mask': (N,) bool  True ha pipeline ok=True,
          'coverage':(N,) float coverage_ratio értékek,
        }
    """
    from src.fretboard import run_v14_pipeline
    from src.hand_landmark import get_landmarker as _get_lm

    if landmarker is None:
        landmarker = _get_lm()

    class_list = sorted(manifest_df["class"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_list)}

    N = len(manifest_df)
    X = np.zeros((N, FEATURE_DIM), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)
    paths = []
    splits = []
    ok_mask = np.zeros(N, dtype=bool)
    coverage_arr = np.zeros(N, dtype=np.float32)

    import builtins
    _orig = builtins.print
    def _silent(*a, **k):
        m = " ".join(str(x) for x in a)
        if any(t in m for t in ["outer_edges", "trapezoid_v9", "nut_detect",
                                  "step7", "fret_fit", "Hough", "klaszter", "bővítve"]):
            return
        _orig(*a, **k)

    for i, (_, row) in enumerate(manifest_df.iterrows()):
        builtins.print = _silent
        r = run_v14_pipeline({"path": row["path"], "class": row["class"]},
                              landmarker=landmarker)
        builtins.print = _orig

        X[i] = assemble_feature_vector(r)
        y[i] = class_to_idx.get(row["class"], 0)
        paths.append(str(row["path"]))
        splits.append(str(row.get("split", "unknown")))
        ok_mask[i] = bool(r.get("ok", False))
        fit = r.get("fit") or {}
        coverage_arr[i] = float(fit.get("coverage_ratio", 0.0))

        # Free memory
        del r

        if verbose and (i + 1) % 50 == 0:
            n_ok = int(ok_mask[:i+1].sum())
            _orig(f"  {i+1}/{N}  ok={n_ok} ({n_ok/(i+1)*100:.0f}%)", flush=True)

    return {
        "X": X,
        "y": y,
        "classes": class_list,
        "paths": paths,
        "splits": splits,
        "ok_mask": ok_mask,
        "coverage": coverage_arr,
    }


def save_features(batch: dict,
                  path: Optional[Path] = None) -> Path:
    """Elmenti a feature mátrixot NPZ formátumban.

    Visszaad: a mentett fájl útvonala.
    """
    if path is None:
        path = PATHS["features_v14"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=batch["X"],
        y=batch["y"],
        classes=np.array(batch["classes"]),
        paths=np.array(batch["paths"]),
        splits=np.array(batch["splits"]),
        ok_mask=batch["ok_mask"],
        coverage=batch["coverage"],
        feature_names=np.array(feature_names()),
    )
    return path


def load_features(path: Optional[Path] = None) -> dict:
    """Betölti a features_v14.npz fájlt.

    Visszaad ugyanolyan dict-et mint extract_features_batch.
    """
    if path is None:
        path = PATHS["features_v14"]
    data = np.load(path, allow_pickle=False)
    return {
        "X": data["X"],
        "y": data["y"],
        "classes": list(data["classes"]),
        "paths": list(data["paths"]),
        "splits": list(data["splits"]),
        "ok_mask": data["ok_mask"],
        "coverage": data["coverage"],
        "feature_names": list(data["feature_names"]),
    }


def extract_ml_features(result: dict) -> dict:
    """Extract ML-ready artifacts from a pipeline result.

    Returns a dict containing:
      - 'cnn_roi': canonical BGR ROI (with hand intact) suitable for CNN input
      - 'finger_rel_pos': np.ndarray (5,) relative finger positions between neighbor frets (0..1 or np.nan)
      - 'profile_masked': 1D np.ndarray normalized profile computed while ignoring masked (hand) rows
    """
    canon = result.get("canon")
    out = {"cnn_roi": None, "finger_rel_pos": None, "profile_masked": None}
    if canon is None:
        return out
    out["cnn_roi"] = canon.copy()

    # Predicted fret x positions
    fit = result.get("fit") or {}
    pred = fit.get("predicted_x", {}) if fit else {}
    pred_xs = sorted([float(v) for v in pred.values()]) if pred else []

    # Finger canonical x positions
    fingertips = result.get("fingertips") or []
    rels = np.full((5,), np.nan, dtype=np.float32)
    # fingertips are returned as list of dicts with 'tip_idx' and 'canon_x'
    tip_map = {fp.get("tip_idx"): fp for fp in fingertips}
    from src.constants import FINGER_TIP_IDX
    for i, tip_idx in enumerate(FINGER_TIP_IDX):
        fp = tip_map.get(tip_idx)
        if fp is None:
            continue
        cx = fp.get("canon_x")
        if cx is None or not pred_xs or len(pred_xs) < 2:
            rels[i] = np.nan
            continue
        # Find neighboring frets
        left = None
        right = None
        for px in pred_xs:
            if px <= cx:
                left = px
            elif px > cx and right is None:
                right = px
        if left is None or right is None or abs(right - left) < 1e-6:
            rels[i] = np.nan
        else:
            rel = float((cx - left) / (right - left))
            rels[i] = float(np.clip(rel, 0.0, 1.0))
    out["finger_rel_pos"] = rels

    # Compute masked intensity profile: ignore rows where hand mask indicates finger/palm
    try:
        gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY).astype(np.float32)
        W = gray.shape[1]
        hand_mask = result.get("hand_mask") or result.get("hand_mask_canon")
        if hand_mask is not None and hand_mask.shape[:2] == gray.shape[:2] and np.count_nonzero(hand_mask) > 0:
            prof = np.zeros((W,), dtype=np.float32)
            for x in range(W):
                col = gray[:, x]
                mask_col = hand_mask[:, x]
                vals = col[mask_col == 0]
                if vals.size == 0:
                    vals = col
                prof[x] = float(np.mean(vals)) if vals.size > 0 else 0.0
        else:
            prof = gray.mean(axis=0).astype(np.float32)
        if prof.max() > 1e-6:
            prof = prof / float(prof.max())
        out["profile_masked"] = prof
    except Exception:
        out["profile_masked"] = None

    return out
