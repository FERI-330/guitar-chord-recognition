"""
src/preprocess.py

Konfigurációvezérelt előfeldolgozási lánc a guitar chord recognition pipeline-hoz.

Plug-and-Play: a run_v14_pipeline(preprocessor=...) paraméterén át cserélhető,
a detektáló logika (FretDetectorInterface) semmit sem tud az előfeldolgozásról.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

from src.config import PREPROCESSING_CONFIG

def preprocess_image_input(raw_bytes: bytes, max_long_edge: int = 0) -> np.ndarray:
    """Feltöltött képbájt → BGR numpy tömb, egységesített előkészítéssel.

    Lépések:
      1. PIL megnyitás (EXIF-tudatos)
      2. ImageOps.exif_transpose — telefon/kamera elforgatás korrekció
      3. Arányőrző kicsinyítés, ha max(w, h) > max_long_edge (csak ha max_long_edge > 0)
      4. PIL RGB → OpenCV BGR konverzió

    Args:
        raw_bytes:     A feltöltött képfájl nyers bájttartalma.
        max_long_edge: Ha > 0 és a kép hosszabb éle ennél nagyobb, arányőrzve kicsinyíti.
                       0 (alapértelmezett) = nincs kicsinyítés, eredeti felbontás megmarad.

    Returns:
        BGR numpy uint8 tömb, amit a pipeline közvetlenül felhasználhat.

    Raises:
        ValueError: Ha a bájttartalom nem érvényes képfájl.
    """
    try:
        pil_img = PILImage.open(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError(f"Érvénytelen képfájl: {exc}") from exc

    pil_img = ImageOps.exif_transpose(pil_img)

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    if max_long_edge > 0:
        w, h = pil_img.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_w = max(1, round(w * scale))
            new_h = max(1, round(h * scale))
            pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)

    img_bgr = cv2.cvtColor(np.asarray(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return img_bgr


class ImagePreprocessor:
    """Konfigurációvezérelt képelőfeldolgozó lánc.

    Lépések (mind kapcsolható a configban):
      1. CLAHE — az LAB tér L csatornájára (színinformáció megmarad)
      2. Gaussian Blur — pipeline-szintű pre-blur (elkülönül a Canny belsőjétől)
      3. Normalizálás — "minmax" (cv2.normalize) vagy "histogram_eq" (equalizeHist L-en)

    Példa:
        prep = ImagePreprocessor()
        img_processed = prep.process(img_bgr)

        # Vizualizációhoz közbenső stádiumok:
        stages = prep.process_stages(img_bgr)
        # stages["original"], stages["clahe"], stages["final"], ...
    """

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config if config is not None else PREPROCESSING_CONFIG
        self._clahe: cv2.CLAHE | None = None
        if self.cfg.get("clahe_enabled", False):
            self._clahe = cv2.createCLAHE(
                clipLimit=float(self.cfg["clahe_clip_limit"]),
                tileGridSize=tuple(self.cfg["clahe_tile_grid_size"]),
            )

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        """Előfeldolgozási lánc alkalmazása. Visszaad: BGR kép."""
        return self.process_stages(img_bgr)["final"]

    def process_stages(self, img_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """Előfeldolgozási lánc lépésenként, vizualizációhoz.

        Visszaad: dict, kulcsok: "original", "clahe" (ha enabled),
        "blur" (ha enabled), "normalized" (ha enabled), "final".
        """
        stages: dict[str, np.ndarray] = {"original": img_bgr.copy()}
        img = img_bgr.copy()

        if self.cfg.get("clahe_enabled", False) and self._clahe is not None:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            stages["clahe"] = img.copy()

        if self.cfg.get("blur_enabled", False):
            k = int(self.cfg["blur_ksize"])
            k = k if k % 2 else k + 1
            img = cv2.GaussianBlur(img, (k, k), 0)
            stages["blur"] = img.copy()

        if self.cfg.get("normalize_enabled", False):
            method = self.cfg.get("normalize_method", "minmax")
            if method == "histogram_eq":
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.equalizeHist(l)
                img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            else:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            stages["normalized"] = img.copy()

        stages["final"] = img.copy()
        return stages
