"""
train_models.py

sklearn (SVM, RF, LR) és CNN modellek tanítása az exportált npy adatokból.

Bemenetek (data/features/ mappa):
  X_basic.npy, X_inlay.npy, X_images.npy, y.npy, splits.npy, class_names.npy

Kimenetek (models/ mappa, verziózva):
  svm_basic_v1.pkl   svm_inlay_v1.pkl
  rf_basic_v1.pkl    rf_inlay_v1.pkl
  lr_basic_v1.pkl    lr_inlay_v1.pkl
  cnn_v1.pth

Futtatás:
    python train_models.py [--no-cnn] [--epochs 20]
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import CFG, PATHS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _next_version(model_dir: Path, stem: str) -> int:
    """Megkeresi a legmagasabb verziószámot és visszaad N+1-et."""
    pattern = re.compile(rf"^{re.escape(stem)}_v(\d+)\.(pkl|pth)$")
    versions = [
        int(m.group(1))
        for f in model_dir.iterdir()
        if (m := pattern.match(f.name))
    ]
    return (max(versions) + 1) if versions else 1


def _split_arrays(X: np.ndarray, y: np.ndarray, splits: np.ndarray):
    """Szétválaszt (X_tr, y_tr), (X_va, y_va), (X_te, y_te) hármasokra."""
    tr = splits == "train"
    va = splits == "val"
    te = splits == "test"
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


# ─────────────────────────────────────────────────────────────────────────────
# sklearn training
# ─────────────────────────────────────────────────────────────────────────────

def _fit_pipeline(name: str, clf: Any,
                  X_tr: np.ndarray, y_tr: np.ndarray,
                  X_va: np.ndarray, y_va: np.ndarray) -> Pipeline:
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_va)
    acc = accuracy_score(y_va, y_pred)
    f1  = f1_score(y_va, y_pred, average="macro", zero_division=0)
    print(f"  val  acc={acc:.3f}  macro_F1={f1:.3f}")
    return pipe


def _save_sklearn(pipe: Pipeline, path: Path, meta: dict) -> None:
    obj = {"model": pipe, **meta}
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  → {path.name}")


def train_sklearn_models(features_dir: Path, model_dir: Path,
                         seed: int = 42) -> None:
    print("\n=== sklearn modellek tanítása ===")
    X_basic = np.load(features_dir / "X_basic.npy")
    X_inlay = np.load(features_dir / "X_inlay.npy")
    y       = np.load(features_dir / "y.npy")
    splits  = np.load(features_dir / "splits.npy", allow_pickle=True)
    classes = list(np.load(features_dir / "class_names.npy", allow_pickle=True))
    print(f"  Adatok: {len(y)} minta, {len(classes)} osztály: {classes}")

    configs: list[tuple[str, np.ndarray, Any]] = [
        ("svm_basic", X_basic,
         SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=seed)),
        ("svm_inlay", X_inlay,
         SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=seed)),
        ("rf_basic",  X_basic,
         RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        ("rf_inlay",  X_inlay,
         RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        ("lr_basic",  X_basic,
         LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
        ("lr_inlay",  X_inlay,
         LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
    ]

    model_dir.mkdir(parents=True, exist_ok=True)
    for stem, X, clf in configs:
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = _split_arrays(X, y, splits)
        print(f"\n[{stem}]  train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")
        pipe = _fit_pipeline(stem, clf, X_tr, y_tr, X_va, y_va)
        v    = _next_version(model_dir, stem)
        path = model_dir / f"{stem}_v{v}.pkl"
        _save_sklearn(pipe, path, {
            "classes":     classes,
            "feature_set": stem.split("_")[1],   # "basic" or "inlay"
        })

    print("\nsklearn modellek mentve.")


# ─────────────────────────────────────────────────────────────────────────────
# CNN training  (MobileNetV3-Small, Phase A → Phase B)
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn_model(features_dir: Path, model_dir: Path,
                    epochs_a: int = 15, epochs_b: int = 15,
                    batch_size: int = 16, seed: int = 42) -> None:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset

    from src.models import build_model
    from src.train import EarlyStopping

    print("\n=== CNN tanítása (MobileNetV3-Small) ===")
    X_images = np.load(features_dir / "X_images.npy")   # (N, H, W, 3) float32 [0,1]
    y_arr    = np.load(features_dir / "y.npy")
    splits   = np.load(features_dir / "splits.npy", allow_pickle=True)
    classes  = list(np.load(features_dir / "class_names.npy", allow_pickle=True))
    n_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}  |  {len(y_arr)} minta, {n_classes} osztály")
    torch.manual_seed(seed)

    # (N, H, W, 3) → (N, 3, H, W)
    X_t = torch.from_numpy(X_images.transpose(0, 3, 1, 2)).float()
    y_t = torch.from_numpy(y_arr).long()

    tr = splits == "train"
    va = splits == "val"

    train_dl = DataLoader(TensorDataset(X_t[tr], y_t[tr]),
                          batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(TensorDataset(X_t[va], y_t[va]),
                          batch_size=batch_size, shuffle=False, num_workers=2)

    model    = build_model("mobilenet_v3_small", num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    model_dir.mkdir(parents=True, exist_ok=True)
    v        = _next_version(model_dir, "cnn")
    ckpt     = model_dir / f"cnn_v{v}.pth"
    stopper  = EarlyStopping(patience=7, ckpt_path=ckpt)

    def _run_epoch(loader, train: bool, opt=None):
        model.train() if train else model.eval()
        total_loss, correct = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total_loss += loss.item()
                correct    += (out.argmax(1) == yb).sum().item()
        n = len(loader.dataset)
        return total_loss / len(loader), correct / n

    # ── Phase A: frozen backbone ──────────────────────────────────────────────
    print("\n  Phase A (frozen backbone)...")
    for p in model.features.parameters():
        p.requires_grad = False
    opt_a = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=CFG["lr_phase_a"])

    for ep in range(1, epochs_a + 1):
        tr_loss, tr_acc = _run_epoch(train_dl, train=True, opt=opt_a)
        va_loss, va_acc = _run_epoch(val_dl,   train=False)
        print(f"    A {ep:>2}/{epochs_a}  tr_acc={tr_acc:.3f}  "
              f"va_loss={va_loss:.4f}  va_acc={va_acc:.3f}")
        if stopper.step(va_loss, model):
            print("    Early stop (Phase A).")
            break
    stopper.restore_best(model)

    # ── Phase B: unfreeze backbone ────────────────────────────────────────────
    print("\n  Phase B (fine-tune backbone)...")
    for p in model.features.parameters():
        p.requires_grad = True
    opt_b = AdamW([
        {"params": model.features.parameters(), "lr": CFG["lr_phase_b_backbone"]},
        {"params": model.classifier.parameters(), "lr": CFG["lr_phase_b_head"]},
    ])
    stopper_b = EarlyStopping(patience=7, ckpt_path=ckpt)

    for ep in range(1, epochs_b + 1):
        tr_loss, tr_acc = _run_epoch(train_dl, train=True, opt=opt_b)
        va_loss, va_acc = _run_epoch(val_dl,   train=False)
        print(f"    B {ep:>2}/{epochs_b}  tr_acc={tr_acc:.3f}  "
              f"va_loss={va_loss:.4f}  va_acc={va_acc:.3f}")
        if stopper_b.step(va_loss, model):
            print("    Early stop (Phase B).")
            break
    stopper_b.restore_best(model)

    torch.save(model.state_dict(), ckpt)
    print(f"\n  CNN mentve → {ckpt.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guitar chord model trainer")
    parser.add_argument("--features-dir", type=Path,
                        default=PATHS["data"] / "features",
                        help="Exportált npy fájlok mappája")
    parser.add_argument("--model-dir", type=Path,
                        default=PATHS["root"] / "models",
                        help="Kimeneti modell mappa")
    parser.add_argument("--no-cnn", action="store_true",
                        help="Csak sklearn modellek tanítása")
    parser.add_argument("--epochs-a", type=int, default=15,
                        help="CNN Phase A epoch szám")
    parser.add_argument("--epochs-b", type=int, default=15,
                        help="CNN Phase B epoch szám")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_sklearn_models(args.features_dir, args.model_dir, seed=args.seed)
    if not args.no_cnn:
        train_cnn_model(args.features_dir, args.model_dir,
                        epochs_a=args.epochs_a, epochs_b=args.epochs_b,
                        seed=args.seed)
    print("\n✓ Kész.")
