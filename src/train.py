"""
src/train.py

Training loop, kiértékelés, early stopping, checkpoint kezelés.
Phase-A / Phase-B protokoll (frozen backbone → unfreeze).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import CFG, PATHS


# ─────────────────────────────────────────────────────────────────────────────
# EarlyStopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Validációs loss alapú early stopping + best model mentés."""

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 ckpt_path: Optional[Path] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.ckpt_path = ckpt_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Visszaad True-t ha le kell állítani."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            if self.ckpt_path is not None:
                self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.best_state, self.ckpt_path)
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"  Best model restored (val_loss={self.best_loss:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Epoch szintű lépések
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    """Egy tanítási epoch.

    Visszaad: (átlagos loss, accuracy)
    """
    model.train()
    total_loss = correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Validáció / teszt kiértékelés.

    Visszaad: (loss, accuracy, all_preds, all_labels)
    """
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels))


# ─────────────────────────────────────────────────────────────────────────────
# Kétfázisú training protokoll
# ─────────────────────────────────────────────────────────────────────────────

def train_two_phase(model: nn.Module,
                    model_name: str,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    class_weights: torch.Tensor,
                    device: torch.device,
                    label: str = "model",
                    verbose: bool = True) -> dict:
    """Phase-A (frozen backbone) → Phase-B (utolsó blokkok felolvadnak).

    Args:
        model: már head-csereált modell (freeze_backbone ELŐTT)
        model_name: 'mobilenet_v3_small', stb. – a freeze/unfreeze segédletekhez
        label: checkpoint fájlnév prefix

    Visszaad: metrics dict (history, best_val_acc stb.)
    """
    from src.models import freeze_backbone, unfreeze_last_blocks

    ckpt_dir = PATHS["checkpoint_dir"]
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Phase A ───────────────────────────────────────────────────────────
    freeze_backbone(model, model_name)
    if verbose:
        from src.models import count_parameters
        _, trainable = count_parameters(model)
        print(f"[Phase A] {label}  trainable={trainable:,}")

    optA = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=CFG["lr_phase_a"], weight_decay=1e-4)
    schA = CosineAnnealingLR(optA, T_max=CFG["epochs_a"], eta_min=1e-5)
    esA = EarlyStopping(patience=CFG["patience"],
                        ckpt_path=ckpt_dir / f"best_{label}_phA.pth")

    hist_a = []
    for ep in range(1, CFG["epochs_a"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optA, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        schA.step()
        hist_a.append(dict(ep=ep, tr_loss=tr_loss, tr_acc=tr_acc,
                           vl_loss=vl_loss, vl_acc=vl_acc, phase="A"))
        if verbose:
            print(f"  A ep{ep:>2}  tr={tr_acc:.3f}  vl={vl_acc:.3f}  vl_loss={vl_loss:.4f}")
        if esA.step(vl_loss, model):
            if verbose:
                print("  Phase A early stop.")
            break
    esA.restore_best(model)

    # ── Phase B ───────────────────────────────────────────────────────────
    unfreeze_last_blocks(model, model_name)
    if verbose:
        from src.models import count_parameters
        _, trainable = count_parameters(model)
        print(f"[Phase B] {label}  trainable={trainable:,}")

    param_groups = [
        {"params": filter(lambda p: p.requires_grad,
                          _head_params(model, model_name)),
         "lr": CFG["lr_phase_b_head"]},
        {"params": filter(lambda p: p.requires_grad,
                          _backbone_params(model, model_name)),
         "lr": CFG["lr_phase_b_backbone"]},
    ]
    optB = AdamW(param_groups, weight_decay=1e-4)
    schB = CosineAnnealingLR(optB, T_max=CFG["epochs_b"], eta_min=1e-6)
    esB = EarlyStopping(patience=CFG["patience"],
                        ckpt_path=ckpt_dir / f"best_{label}_phB.pth")

    hist_b = []
    for ep in range(1, CFG["epochs_b"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optB, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        schB.step()
        hist_b.append(dict(ep=ep, tr_loss=tr_loss, tr_acc=tr_acc,
                           vl_loss=vl_loss, vl_acc=vl_acc, phase="B"))
        if verbose:
            print(f"  B ep{ep:>2}  tr={tr_acc:.3f}  vl={vl_acc:.3f}  vl_loss={vl_loss:.4f}")
        if esB.step(vl_loss, model):
            if verbose:
                print("  Phase B early stop.")
            break
    esB.restore_best(model)

    best_val_acc = max(h["vl_acc"] for h in hist_a + hist_b)
    return {
        "history": hist_a + hist_b,
        "best_val_acc": best_val_acc,
        "ckpt_phA": ckpt_dir / f"best_{label}_phA.pth",
        "ckpt_phB": ckpt_dir / f"best_{label}_phB.pth",
    }


def _head_params(model: nn.Module, name: str):
    """Iterál a classifier/fc paramétereken."""
    if name in ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "efficientnet_b3"):
        return model.classifier.parameters()
    elif name == "shufflenet_v2":
        return list(model.fc.parameters()) + list(model.conv5.parameters())
    elif name == "resnet50":
        return model.fc.parameters()
    return model.parameters()


def _backbone_params(model: nn.Module, name: str):
    """Iterál az összes NEM-head paraméteren (a felolvadt backbone részek)."""
    if name in ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "efficientnet_b3"):
        return model.features.parameters()
    elif name == "shufflenet_v2":
        return model.stage4.parameters()
    elif name == "resnet50":
        return model.layer4.parameters()
    return iter([])


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint betöltés
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(model: nn.Module,
                    ckpt_path: Path,
                    device: torch.device) -> nn.Module:
    """Checkpoint betöltése modellbe."""
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model
