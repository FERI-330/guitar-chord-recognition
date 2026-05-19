"""
app.py – Streamlit demo: Gitár Akkord Felismerő

Kizárólag UI-logika. Minden üzleti logika src/inference.py-ban van.

Indítás:
    streamlit run app.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PATHS
from src.inference import CLASS_NAMES, InferenceResult, load_cnn, load_svm, predict

# ─── Oldal-konfiguráció ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gitár Akkord Felismerő",
    page_icon="🎸",
    layout="wide",
)

# ─── Modellek betöltése (egyszer, cache-elve) ─────────────────────────────────
@st.cache_resource(show_spinner="CNN modell betöltése...")
def get_cnn():
    return load_cnn()

@st.cache_resource(show_spinner="SVM modell betöltése...")
def get_svm():
    return load_svm()

# ─── Inferencia cache (kép + modell kombinációnként) ─────────────────────────
@st.cache_data(show_spinner=False)
def run_predict(image_bytes: bytes, use_cnn: bool) -> InferenceResult:
    img_array = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return predict(
        image_bgr,
        cnn_model=get_cnn() if use_cnn else None,
        svm_model=get_svm() if not use_cnn else None,
    )

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Beállítások")
    model_choice = st.radio(
        "Osztályozó modell",
        [
            "CNN – MobileNetV3-Large (97.8% test acc)",
            "SVM – 42-dim features (91.1% test acc)",
        ],
    )
    st.divider()
    st.caption("**Pipeline:** V14 (Hough + homográfia + 17.817 szabály)")
    st.caption("**Osztályok:** " + ", ".join(CLASS_NAMES))

use_cnn = model_choice.startswith("CNN")

# ─── Fejléc ───────────────────────────────────────────────────────────────────
st.title("🎸 Gitár Akkord Felismerő")
st.caption(
    "Tölts fel egy képet a gitárnyakról — a rendszer detektálja a fogólapot "
    "és azonosítja a játszott akkordot."
)

# ─── Feltöltő ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Kép feltöltése (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("Tölts fel egy képet a fenti gombbal az eredmény megtekintéséhez.")
    st.stop()

image_bytes = uploaded.getbuffer().tobytes()

with st.spinner("Pipeline futtatás és osztályozás..."):
    result = run_predict(image_bytes, use_cnn)

# ─── Képek egymás mellett ────────────────────────────────────────────────────
col_orig, col_overlay = st.columns(2)

img_array = np.frombuffer(image_bytes, np.uint8)
original_rgb = cv2.cvtColor(
    cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
)
overlay_rgb = result.overlay_bgr[:, :, ::-1]

with col_orig:
    st.image(original_rgb, caption="Eredeti kép", use_container_width=True)

with col_overlay:
    if result.ok:
        caption = f"Fretboard overlay  |  coverage {result.coverage:.0%}"
    else:
        reason = result.pipeline_result.get("invalid_reason", "ismeretlen")
        caption = f"Pipeline FAIL – {reason}"
    st.image(overlay_rgb, caption=caption, use_container_width=True)

# ─── Eredmény metrikák ────────────────────────────────────────────────────────
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Akkord", result.chord)
m2.metric("Confidence", f"{result.confidence:.1%}")
m3.metric("Pipeline", "OK ✓" if result.ok else "FAIL ✗")

# ─── Top-3 sávdiagram (CNN esetén) ───────────────────────────────────────────
if use_cnn and len(result.top3) > 1:
    st.divider()
    st.subheader("Top-3 valószínűség")
    for cls_name, prob in result.top3:
        col_lbl, col_bar = st.columns([1, 5])
        col_lbl.write(f"**{cls_name}**")
        col_bar.progress(float(prob), text=f"{prob:.1%}")
