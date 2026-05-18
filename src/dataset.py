"""
src/dataset.py

PyTorch Dataset, DataLoader, transzformációk, class-weight számítás.

Két Dataset típus:
  ManifestDataset – nyers kép → CNN fine-tuninghoz
  FeatureDataset  – features_v14.npz → ML baseline-hoz
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.config import CFG, PATHS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# Transzformációk
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str):
    """Visszaad egy torchvision transzformációt a megadott split-hez.

    split: 'train' → adatbővítéssel, 'val'/'test' → csak resize+normalize.
    """
    import torchvision.transforms as T
    img_size = CFG["img_size"]
    if split == "train":
        return T.Compose([
            T.Resize(int(img_size * 256 / 224)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 256 / 224)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# ManifestDataset – kép-alapú CNN-hez
# ─────────────────────────────────────────────────────────────────────────────

class ManifestDataset(Dataset):
    """Manifest CSV sor → (kép tensor, osztálycímke int) pár."""

    def __init__(self,
                 df: pd.DataFrame,
                 class_to_idx: dict,
                 transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row["class"]]
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# FeatureDataset – NPZ-alapú ML-hez
# ─────────────────────────────────────────────────────────────────────────────

class FeatureDataset(Dataset):
    """features_v14.npz → (feature vektor tensor, osztálycímke int) pár."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Class-weight számítás
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y_train: np.ndarray,
                          n_classes: int,
                          device: Optional[torch.device] = None) -> torch.Tensor:
    """Egységes osztálysúly képlet: total / (n_classes × count_per_class).

    Ha egy osztály nem szerepel a train split-ben → súly = 0 (nem büntetjük
    a modellt, de figyelmeztetést adunk).
    """
    total = len(y_train)
    weights = []
    for c in range(n_classes):
        count = int((y_train == c).sum())
        if count == 0:
            print(f"  [class_weights] figyelmeztetés: osztály {c} hiányzik a train split-ből")
            weights.append(0.0)
        else:
            weights.append(total / (n_classes * count))
    w = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        w = w.to(device)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# get_dataloaders – kép-alapú CNN pipeline
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(manifest_path: Optional[Path] = None,
                    batch_size: Optional[int] = None,
                    num_workers: int = 0,
                    device: Optional[torch.device] = None
                    ) -> tuple[DataLoader, DataLoader, DataLoader, list, torch.Tensor]:
    """Manifest CSV-ből létrehozza a train/val/test DataLoader-eket CNN-hez.

    Visszaad: (train_loader, val_loader, test_loader, class_list, class_weights)
    """
    if manifest_path is None:
        manifest_path = PATHS["manifest"]
    if batch_size is None:
        batch_size = CFG["batch_size"]

    df = pd.read_csv(manifest_path)
    class_list = sorted(df["class"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_list)}

    splits = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].reset_index(drop=True)
        tf = get_transforms(split)
        ds = ManifestDataset(sub, class_to_idx, transform=tf)
        shuffle = (split == "train")
        splits[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=(device is not None),
            drop_last=(split == "train"),
        )

    train_df = df[df["split"] == "train"]
    y_train = train_df["class"].map(class_to_idx).values.astype(np.int64)
    weights = compute_class_weights(y_train, len(class_list), device)

    return splits["train"], splits["val"], splits["test"], class_list, weights


def get_feature_dataloaders(npz_path: Optional[Path] = None,
                             batch_size: Optional[int] = None,
                             device: Optional[torch.device] = None
                             ) -> tuple[DataLoader, DataLoader, DataLoader, list, torch.Tensor]:
    """features_v14.npz-ből létrehozza a train/val/test DataLoader-eket feature ML-hez.

    Visszaad: (train_loader, val_loader, test_loader, class_list, class_weights)
    """
    from src.features import load_features
    if npz_path is None:
        npz_path = PATHS["features_v14"]
    if batch_size is None:
        batch_size = CFG["batch_size"]

    data = load_features(npz_path)
    X, y = data["X"], data["y"]
    splits_arr = np.array(data["splits"])
    class_list = data["classes"]

    loaders = {}
    for split in ("train", "val", "test"):
        mask = splits_arr == split
        ds = FeatureDataset(X[mask], y[mask])
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=(device is not None),
        )

    y_train = y[splits_arr == "train"]
    weights = compute_class_weights(y_train, len(class_list), device)

    return loaders["train"], loaders["val"], loaders["test"], class_list, weights
