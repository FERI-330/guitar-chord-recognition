from torchvision import models
import torch.nn as nn


def get_efficientnet_b0(num_classes=8, pretrained=True):
    """Return EfficientNet-B0 with replaced classifier for `num_classes`."""
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        model = models.efficientnet_b0(pretrained=pretrained)

    if hasattr(model, "classifier"):
        in_features = None
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                in_features = m.in_features
        if in_features is None:
            try:
                in_features = model.classifier[1].in_features
            except Exception:
                raise RuntimeError("Could not determine classifier in_features")
        model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, num_classes))
    else:
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unknown model head layout; cannot replace classifier")

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
