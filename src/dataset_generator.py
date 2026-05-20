"""
src/dataset_generator.py

ML-ready dataset export a V14 pipeline segítségével.

Kimenet (data/features/ mappába):
  X_basic.npy    – (N, 56)           float32  assemble_feature_vector
  X_inlay.npy    – (N, 60)           float32  X_basic + 4 inlay x-pozíció
  X_images.npy   – (N, 224, 224, 3)  float32  normalizált RGB képek
  y.npy           – (N,)              int64    osztálycímke-index
  splits.npy      – (N,)              str      'train'/'val'/'test'
  class_names.npy – (K,)              str      index → osztálynév

Inlay feature (4 dim): bund 3, 5, 7, 9 egypontos inlay-ek normalizált
  x-pozíciója a kanonikus képben [0,1]. Ha fretboard_detected=0 → 0.0.

CLI:
    python -m src.dataset_generator [--output-dir data/features]
"""
from __future__ import annotations

import argparse
import builtins
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PATHS
from src.constants import CANONICAL_W, FRET_POS_NORM
from src.features import FEATURE_DIM, get_ml_ready_payload

# Egypontos inlay bundok (12-es kettős pontot kihagyva → 4 dim)
_INLAY_4 = [3, 5, 7, 9]
INLAY_DIM = len(_INLAY_4)          # 4
FEATURE_DIM_INLAY = FEATURE_DIM + INLAY_DIM  # 60

# Pipeline debug szövegek, amiket elnyomunk a batch futtatás során
_SUPPRESS_TOKENS = frozenset({
    "outer_edges", "trapezoid_v9", "nut_detect", "step7",
    "fret_fit", "Hough", "klaszter", "bővítve",
})


def _inlay_features(result: dict) -> np.ndarray:
    """Bund 3/5/7/9 normalizált x-pozíció a fit kimenetéből.

    Ha a bund szerepel a fit['predicted_x']-ben, azt használja.
    Különben a FRET_POS_NORM elméletit veszi alapul.
    Fretboard_detected=0 esetén csupa 0.
    """
    vec = np.zeros(INLAY_DIM, dtype=np.float32)
    ok = bool(result.get("ok", False))
    if not ok:
        return vec
    fit = result.get("fit") or {}
    coverage = float(fit.get("coverage_ratio", 0.0))
    if coverage < 0.40:
        return vec
    pred_x: dict = fit.get("predicted_x") or {}
    w = float(CANONICAL_W)
    for i, fret_n in enumerate(_INLAY_4):
        if fret_n in pred_x:
            vec[i] = float(np.clip(float(pred_x[fret_n]) / w, 0.0, 1.0))
        else:
            vec[i] = float(np.clip(FRET_POS_NORM[fret_n], 0.0, 1.0))
    return vec


def export_dataset(
    manifest_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    img_size: tuple[int, int] = (224, 224),
    verbose: bool = True,
) -> Path:
    """Pipeline futtatása minden képre és ML adatok exportálása.

    Args:
        manifest_path: CSV útvonal (None → PATHS['manifest']).
        output_dir:    Kimeneti mappa (None → data/features/).
        img_size:      (W, H) célméret a CNN képekhez.
        verbose:       Haladási kiírás 25 képenként.

    Returns:
        output_dir Path ahol az npy fájlok vannak.
    """
    from src.fretboard import run_v14_pipeline
    from src.hand_landmark import get_landmarker

    if manifest_path is None:
        manifest_path = PATHS["manifest"]
    if output_dir is None:
        output_dir = PATHS["data"] / "features"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df["path"] = df["path"].astype(str)

    class_list = sorted(df["class"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    N = len(df)

    X_basic  = np.zeros((N, FEATURE_DIM),       dtype=np.float32)
    X_inlay  = np.zeros((N, FEATURE_DIM_INLAY),  dtype=np.float32)
    X_images = np.zeros((N, img_size[1], img_size[0], 3), dtype=np.float32)
    y        = np.zeros(N, dtype=np.int64)
    splits   = np.empty(N, dtype=object)

    landmarker = get_landmarker()
    _orig_print = builtins.print

    def _silent(*a, **k):
        msg = " ".join(str(x) for x in a)
        if any(t in msg for t in _SUPPRESS_TOKENS):
            return
        _orig_print(*a, **k)

    for i, row in df.iterrows():
        builtins.print = _silent
        result = run_v14_pipeline(
            {"path": row["path"], "class": row["class"]},
            landmarker=landmarker,
        )
        builtins.print = _orig_print

        payload = get_ml_ready_payload(result, target_size=img_size)

        X_basic[i]  = payload["feature_vec"]
        X_inlay[i]  = np.concatenate([payload["feature_vec"], _inlay_features(result)])
        X_images[i] = payload["image"]
        y[i]        = class_to_idx.get(str(row["class"]), 0)
        splits[i]   = str(row.get("split", "unknown"))

        del result, payload

        if verbose and (i + 1) % 25 == 0:
            _orig_print(f"  {i+1}/{N} feldolgozva", flush=True)

    np.save(output_dir / "X_basic.npy",   X_basic)
    np.save(output_dir / "X_inlay.npy",   X_inlay)
    np.save(output_dir / "X_images.npy",  X_images)
    np.save(output_dir / "y.npy",         y)
    np.save(output_dir / "splits.npy",    splits)
    np.save(output_dir / "class_names.npy", np.array(class_list))

    if verbose:
        _orig_print(f"\nExport kész → {output_dir}")
        _orig_print(f"  X_basic:      {X_basic.shape}")
        _orig_print(f"  X_inlay:      {X_inlay.shape}")
        _orig_print(f"  X_images:     {X_images.shape}")
        _orig_print(f"  y:            {y.shape}  ({len(class_list)} osztály: {class_list})")
        n_ok = int((X_basic[:, 43] > 0.5).sum())  # D_fretboard_detected index
        _orig_print(f"  Fretboard OK: {n_ok}/{N} ({n_ok/N*100:.1f}%)")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guitar chord dataset generator")
    parser.add_argument("--manifest",    type=Path, default=None,
                        help="Manifest CSV útvonal")
    parser.add_argument("--output-dir",  type=Path, default=None,
                        help="Kimeneti mappa (alapértelmezés: data/features/)")
    parser.add_argument("--img-size",    type=int, nargs=2, default=[224, 224],
                        metavar=("W", "H"), help="CNN képméret")
    args = parser.parse_args()
    export_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        img_size=tuple(args.img_size),
    )
