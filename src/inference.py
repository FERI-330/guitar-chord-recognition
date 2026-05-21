"""
src/inference.py

Orchestration layer: V14 pipeline + chord classification in a single call.
app.py imports exclusively from here — never touches pipeline/model modules directly.

Public API:
  CLASS_NAMES         – ['A','B','C','D','E','F','G','No hand']
  InferenceResult     – dataclass returned by predict()
  load_cnn(path)      – load MobileNetV3-Large checkpoint
  load_svm(path)      – load scikit-learn SVM checkpoint
  predict(image_bgr, cnn_model, svm_model) → InferenceResult
"""
from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from src.config import CFG, PATHS
from src.dataset import get_transforms
from src.features import assemble_feature_vector
from src.fretboard import run_v14_pipeline
from src.models import build_model
from src.viz import PipelineVisualizer

# Sorted class list – must match the order used during training
# (same as sorted(manifest['class'].unique()), see src/dataset.py)
CLASS_NAMES: list[str] = ["A", "B", "C", "D", "E", "F", "G", "No hand"]


@dataclass
class InferenceResult:
    chord: str                         # predicted class label
    confidence: float                  # [0, 1] – softmax max or SVM prob
    top3: list[tuple[str, float]]      # top-3 (class, prob); single item for SVM
    ok: bool                           # V14 pipeline success flag
    coverage: float                    # fret fit coverage_ratio (0 if ok=False)
    pipeline_result: dict              # raw run_v14_pipeline output (img stripped)
    overlay_bgr: np.ndarray            # fretboard + landmark overlay image


def load_cnn(
    path: Optional[Path] = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load MobileNetV3-Large phase-B checkpoint (default: best_mobilenet_v3_large_phB.pth)."""
    if path is None:
        path = PATHS["checkpoint_dir"] / "best_mobilenet_v3_large_phB.pth"
    model = build_model("mobilenet_v3_large", num_classes=CFG["num_classes"])
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_svm(path: Optional[Path] = None) -> Any:
    """Load scikit-learn SVM checkpoint (default: best_ml_model.pkl).

    The pickle stores a dict {'model': Pipeline, 'classes': [...], ...};
    this function returns only the sklearn Pipeline ready for predict().
    """
    if path is None:
        path = PATHS["checkpoint_dir"] / "best_ml_model.pkl"
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"] if isinstance(obj, dict) else obj


def predict(
    image_bgr: np.ndarray,
    cnn_model: Optional[torch.nn.Module] = None,
    svm_model: Optional[Any] = None,
) -> InferenceResult:
    """Run full inference: V14 geometry pipeline + chord classification.

    Exactly one of cnn_model / svm_model must be provided.
    CNN takes priority if both are supplied.

    Returns InferenceResult with chord, confidence, overlay image, and
    pipeline diagnostics.
    """
    if cnn_model is None and svm_model is None:
        raise ValueError("Provide at least one of cnn_model or svm_model.")

    # 1. Save to a lossless temp file — run_v14_pipeline requires a file path.
    # PNG avoids JPEG re-encoding artefacts that can shift MediaPipe landmarks.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        cv2.imwrite(str(tmp_path), image_bgr)

    try:
        result = run_v14_pipeline({"path": str(tmp_path), "class": "?"})
    finally:
        tmp_path.unlink(missing_ok=True)

    # 2. Build overlay (before stripping img from result)
    viz = PipelineVisualizer()
    overlay = viz.draw_fretboard_overlay(image_bgr, result)
    if result.get("landmarks"):
        overlay = viz.draw_landmarks(overlay, result["landmarks"])

    # Strip the large raw image so InferenceResult is serialisation-friendly
    result.pop("img", None)

    coverage = float((result.get("fit") or {}).get("coverage_ratio") or 0.0)

    # 3. Classify
    if cnn_model is not None:
        chord, confidence, top3 = _classify_cnn(image_bgr, cnn_model)
    else:
        chord, confidence, top3 = _classify_svm(result, svm_model)

    # Suppress confidence when the geometry pipeline failed
    if not result.get("ok"):
        confidence = 0.0

    return InferenceResult(
        chord=chord,
        confidence=confidence,
        top3=top3,
        ok=result.get("ok", False),
        coverage=coverage,
        pipeline_result=result,
        overlay_bgr=overlay,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_cnn(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
) -> tuple[str, float, list[tuple[str, float]]]:
    from PIL import Image as PILImage
    transform = get_transforms("val")
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(PILImage.fromarray(img_rgb)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]
    return top3[0][0], top3[0][1], top3


def _classify_svm(
    pipeline_result: dict,
    svm_model: Any,
) -> tuple[str, float, list[tuple[str, float]]]:
    # SVM was trained on Group B features only (first 42 of the 56-dim vector)
    from src.features import GROUP_B_SIZE
    feat = assemble_feature_vector(pipeline_result)[:GROUP_B_SIZE].reshape(1, -1)
    pred = svm_model.predict(feat)[0]
    # Model predicts integer class indices matching CLASS_NAMES order
    chord = CLASS_NAMES[int(pred)] if isinstance(pred, (int, float, np.integer)) else str(pred)
    try:
        confidence = float(svm_model.predict_proba(feat)[0].max())
    except AttributeError:
        confidence = 1.0
    return chord, confidence, [(chord, confidence)]
