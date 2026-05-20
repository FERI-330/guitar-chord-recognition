"""
src/augmentor.py

Offline adataugmentáció az alulreprezentált osztályok kiegyenlítéséhez.

Csak a TRAIN split-et augmentálja – a val/test érintetlen marad.

Augmentációs műveletek (kombinálva, véletlenszerűen):
  • Elforgatás        ±ROTATE_MAX fok
  • Fényerő jitter    [BRIGHT_LO, BRIGHT_HI]
  • Kontraszt jitter  [CONTR_LO, CONTR_HI]
  • Gaussian zaj      σ ∈ [NOISE_LO, NOISE_HI]
  • Gaussian blur     kernel ∈ {0, 3, 5}
  • Perspektíva warp  enyhe sarokelmozdulás
  • Véletlen crop     [CROP_LO, 1.0] × eredeti méret + resize

Kimenet:
  data/augmented/{osztály}/{eredeti_stem}_aug{n}.jpg

Gyors API:
    from src.augmentor import run_augmentation
    new_manifest = run_augmentation(target_per_class=50)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from src.config import PATHS

# ─── Augmentációs paraméterek ────────────────────────────────────────────────
ROTATE_MAX = 18       # fok
BRIGHT_LO  = 0.65
BRIGHT_HI  = 1.45
CONTR_LO   = 0.75
CONTR_HI   = 1.30
NOISE_LO   = 4.0
NOISE_HI   = 22.0
CROP_LO    = 0.82     # a kép melyik hányadát tartsuk meg min.
PERSP_MAX  = 0.03     # relatív sarokelmozdulás max.
JPEG_Q     = 92       # JPEG mentési minőség


# ─── Egyedi transzformációk ───────────────────────────────────────────────────

def _rotate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(-ROTATE_MAX, ROTATE_MAX)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT_101)


def _brightness(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    factor = rng.uniform(BRIGHT_LO, BRIGHT_HI)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _contrast(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    factor = rng.uniform(CONTR_LO, CONTR_HI)
    mean   = float(img.mean())
    return np.clip((img.astype(np.float32) - mean) * factor + mean,
                   0, 255).astype(np.uint8)


def _noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(NOISE_LO, NOISE_HI)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.choice([0, 3, 5]))
    if k == 0:
        return img
    return cv2.GaussianBlur(img, (k, k), 0)


def _perspective(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    d    = PERSP_MAX
    src  = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Minden sarokpontot véletlenszerűen tolunk el
    offsets = rng.uniform(-d, d, (4, 2)) * np.array([w, h], dtype=np.float32)
    dst  = np.clip(src + offsets, 0, [w - 1, h - 1]).astype(np.float32)
    M    = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT_101)


def _crop_resize(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w  = img.shape[:2]
    scale = rng.uniform(CROP_LO, 1.0)
    ch, cw = int(h * scale), int(w * scale)
    y0 = int(rng.uniform(0, h - ch))
    x0 = int(rng.uniform(0, w - cw))
    cropped = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ─── Egy kép augmentálása ─────────────────────────────────────────────────────

# Prioritás-súlyok az egyes műveletek aktiválásához (mindig ≥ 1 aktív)
_OPS = [_rotate, _brightness, _contrast, _noise, _blur, _perspective, _crop_resize]
_PROB = [0.75,   0.80,        0.70,      0.60,   0.50,  0.50,         0.65]


def augment_one(img: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """Véletlenszerű kombinációjú augmentáció egy képen.

    Legalább 2 művelet mindig aktiválódik.
    """
    active = [op for op, p in zip(_OPS, _PROB) if rng.random() < p]
    if len(active) < 2:                       # biztosítjuk a minimumot
        active = list(rng.choice(_OPS, size=2, replace=False))  # type: ignore[arg-type]
    rng.shuffle(active)                       # véletlenszerű sorrend
    result = img.copy()
    for op in active:
        result = op(result, rng)
    return result


# ─── Osztályszintű augmentáció ────────────────────────────────────────────────

def augment_class(
    class_name: str,
    source_paths: list[Path],
    n_needed: int,
    output_dir: Path,
    rng: np.random.Generator,
) -> list[Path]:
    """Egy osztályhoz generál `n_needed` augmentált képet.

    Körkörösen végigmegy a forrásképeken, minden képből legfeljebb
    (n_needed // len(sources) + 2) augmentált változatot készít.

    Returns:
        Elmentett augmentált képek útvonalainak listája.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    src_cycle = list(source_paths)
    rng.shuffle(src_cycle)

    src_idx  = 0
    aug_idx  = 0

    while len(saved) < n_needed:
        src_path = src_cycle[src_idx % len(src_cycle)]
        src_idx += 1

        img = cv2.imread(str(src_path))
        if img is None:
            continue

        aug_img  = augment_one(img, rng)
        out_name = f"{src_path.stem}_aug{aug_idx:04d}.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), aug_img,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        saved.append(out_path)
        aug_idx += 1

    return saved


# ─── Fő belépési pont ─────────────────────────────────────────────────────────

def run_augmentation(
    manifest_path: Optional[Path] = None,
    output_root:   Optional[Path] = None,
    target_per_class: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Augmentált képeket generál, és visszaad egy kibővített manifest DataFrame-et.

    Csak a TRAIN split képeit augmentálja.
    Az eredeti manifest sorait megtartja; az új sorokhoz `source='augmented'` kerül.

    Args:
        manifest_path:    Forrás manifest CSV (None → PATHS['manifest']).
        output_root:      Kimeneti mappa (None → data/augmented/).
        target_per_class: Cél darabszám per osztály a TRAIN splitben.
        seed:             Reprodukálhatóság.

    Returns:
        pd.DataFrame – az eredeti + augmentált sorok.
        A visszaadott DataFrame automatikusan mentésre kerül
        `data/split_manifest_aug.csv` névvel.
    """
    if manifest_path is None:
        manifest_path = PATHS["manifest"]
    if output_root is None:
        output_root = PATHS["data"] / "augmented"

    df  = pd.read_csv(manifest_path)
    rng = np.random.default_rng(seed)

    train_df   = df[df["split"] == "train"].copy()
    class_list = sorted(df["class"].unique().tolist())

    # ── Osztályonkénti darabszám és hiány kiszámítása ─────────────────────────
    counts = train_df["class"].value_counts().to_dict()
    print(f"\n{'Osztály':10s}  {'Meglévő':>8s}  {'Hiány':>6s}  {'Akció'}")
    print("─" * 45)
    aug_rows: list[dict] = []

    for cls in class_list:
        have   = counts.get(cls, 0)
        needed = max(0, target_per_class - have)
        if needed == 0:
            print(f"  {cls:8s}  {have:8d}  {needed:6d}  (kihagyva)")
            continue

        print(f"  {cls:8s}  {have:8d}  {needed:6d}  → augmentálás...")
        src_paths = [Path(p) for p in train_df[train_df["class"] == cls]["path"]]
        out_dir   = output_root / cls

        new_paths = augment_class(cls, src_paths, needed, out_dir, rng)

        for p in new_paths:
            aug_rows.append({
                "split":    "train",
                "class":    cls,
                "filename": p.name,
                "path":     str(p),
                "size_kb":  round(p.stat().st_size / 1024, 2),
                "source":   "augmented",
            })

    # ── Eredmény manifest összeállítása ───────────────────────────────────────
    if "source" not in df.columns:
        df["source"] = "original"

    aug_df  = pd.DataFrame(aug_rows)
    full_df = pd.concat([df, aug_df], ignore_index=True)

    out_csv = PATHS["data"] / "split_manifest_aug.csv"
    full_df.to_csv(out_csv, index=False)
    print(f"\nAugmentált manifest mentve → {out_csv.name}")
    print(f"Sorok: {len(df)} eredeti + {len(aug_df)} augmentált = {len(full_df)} összesen")

    # ── Új eloszlás kiírása ───────────────────────────────────────────────────
    print(f"\n{'Osztály':10s}  {'Train':>6s}  {'Val':>5s}  {'Test':>5s}  {'Total':>6s}")
    print("─" * 42)
    for cls in class_list:
        sub  = full_df[full_df["class"] == cls]
        tr   = (sub["split"] == "train").sum()
        va   = (sub["split"] == "val").sum()
        te   = (sub["split"] == "test").sum()
        flag = " ✱" if cls in [c for c in class_list
                                 if counts.get(c, 0) < target_per_class] else ""
        print(f"  {cls:8s}  {tr:6d}  {va:5d}  {te:5d}  {tr+va+te:6d}{flag}")

    return full_df
