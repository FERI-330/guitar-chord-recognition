"""
src/models.py

CNN architektúra builder-ek és classifier head csere.

Használat:
    model = build_model("mobilenet_v3_small", num_classes=8)
    model = build_model("efficientnet_b0", num_classes=8)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.config import CFG

_SUPPORTED = [
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b3",
    "shufflenet_v2",
    "resnet50",
]


def build_model(name: str,
                num_classes: Optional[int] = None) -> nn.Module:
    """CNN model builder.

    Args:
        name: modell neve (lásd _SUPPORTED lista)
        num_classes: osztályszám; ha None → CFG["num_classes"]

    Visszaad: ImageNet-pretrained model cserélt classifierrel.
    """
    if num_classes is None:
        num_classes = CFG["num_classes"]

    builders = {
        "mobilenet_v3_small": _get_mobilenet_v3_small,
        "mobilenet_v3_large": _get_mobilenet_v3_large,
        "efficientnet_b0":    _get_efficientnet_b0,
        "efficientnet_b3":    _get_efficientnet_b3,
        "shufflenet_v2":      _get_shufflenet_v2,
        "resnet50":           _get_resnet50,
    }
    if name not in builders:
        raise ValueError(f"Ismeretlen modell: '{name}'. Elérhető: {_SUPPORTED}")
    return builders[name](num_classes)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Visszaad: (összes paraméter, tanítható paraméter)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Backbone builder-ek ────────────────────────────────────────────────────

def _get_mobilenet_v3_small(num_classes: int) -> nn.Module:
    m = tv_models.mobilenet_v3_small(
        weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


def _get_mobilenet_v3_large(num_classes: int) -> nn.Module:
    m = tv_models.mobilenet_v3_large(
        weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


def _get_efficientnet_b0(num_classes: int) -> nn.Module:
    m = tv_models.efficientnet_b0(
        weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


def _get_efficientnet_b3(num_classes: int) -> nn.Module:
    m = tv_models.efficientnet_b3(
        weights=tv_models.EfficientNet_B3_Weights.DEFAULT)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


def _get_shufflenet_v2(num_classes: int) -> nn.Module:
    m = tv_models.shufflenet_v2_x1_0(
        weights=tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _get_resnet50(num_classes: int) -> nn.Module:
    m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# ── Phase-A/B fagyasztás segédletek ───────────────────────────────────────

def freeze_backbone(model: nn.Module, name: str) -> None:
    """Fázis A: csak a classifier head tanítható."""
    _freeze_all(model)
    _unfreeze_head(model, name)


def unfreeze_last_blocks(model: nn.Module, name: str) -> None:
    """Fázis B: utolsó blokkok + head felolvadnak."""
    _freeze_all(model)
    _unfreeze_head(model, name)
    _unfreeze_last_blocks(model, name)


def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_head(model: nn.Module, name: str) -> None:
    if name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif name in ("efficientnet_b0", "efficientnet_b3"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif name == "shufflenet_v2":
        for p in model.fc.parameters():
            p.requires_grad = True
        for p in model.conv5.parameters():
            p.requires_grad = True
    elif name == "resnet50":
        for p in model.fc.parameters():
            p.requires_grad = True


def _unfreeze_last_blocks(model: nn.Module, name: str) -> None:
    if name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        # utolsó 3 features blokk
        feats = list(model.features.children())
        for block in feats[-3:]:
            for p in block.parameters():
                p.requires_grad = True
    elif name in ("efficientnet_b0", "efficientnet_b3"):
        feats = list(model.features.children())
        for block in feats[-3:]:
            for p in block.parameters():
                p.requires_grad = True
    elif name == "shufflenet_v2":
        for p in model.stage4.parameters():
            p.requires_grad = True
    elif name == "resnet50":
        for p in model.layer4.parameters():
            p.requires_grad = True


# Opcionális típusjelzés a build_model-ban
from typing import Optional  # noqa: E402 – placed here to avoid circular
