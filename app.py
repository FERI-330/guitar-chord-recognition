"""
app.py – Streamlit demo: Gitár Akkord Felismerő

UI-logika. Minden pipeline-logika src/inference.py-ban van, hogy az
eredmények 100%-ban azonosak legyenek a notebook run_v14_pipeline
kimenetével.

Módok:
  Diagnostic OFF → csak canon_norm kép + akkord neve
  Diagnostic ON  → 16 paneles viz_diagnostics audit nézet

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
from src.preprocess import preprocess_image_input

# Max long-edge (px) for uploaded images — resizes 4K shots to a manageable size
# while preserving aspect ratio. Pipeline thresholds tolerate this resolution.
_UPLOAD_MAX_PX = 1920

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
    image_bgr = preprocess_image_input(image_bytes, max_long_edge=_UPLOAD_MAX_PX)
    return predict(
        image_bgr,
        cnn_model=get_cnn() if use_cnn else None,
        svm_model=get_svm() if not use_cnn else None,
    )


@st.cache_data(show_spinner=False)
def decode_for_display(image_bytes: bytes) -> np.ndarray:
    """Bytes → RGB numpy array a Streamlit st.image() számára."""
    bgr = preprocess_image_input(image_bytes, max_long_edge=_UPLOAD_MAX_PX)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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

    diagnostic_mode = st.toggle(
        "Diagnostic Mode",
        value=False,
        help=(
            "BE: 16-panel pipeline audit (előkészítés, geometria, "
            "detekció, eredmény).\n\n"
            "KI: Csak a kanonikus ROI kép és az akkord neve."
        ),
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

# ─── Eredmény metrikák (mindig látható) ───────────────────────────────────────
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Akkord", result.chord)
m2.metric("Confidence", f"{result.confidence:.1%}")
m3.metric("Pipeline", "OK ✓" if result.ok else "FAIL ✗")

# ─── DIAGNOSTIC MODE OFF → egyszerű nézet ────────────────────────────────────
if not diagnostic_mode:
    st.divider()

    canon_norm = result.pipeline_result.get("canon_norm")
    if canon_norm is not None:
        canon_rgb = cv2.cvtColor(canon_norm, cv2.COLOR_BGR2RGB)
        st.image(
            canon_rgb,
            caption=f"Kanonikus ROI (nut-bal)  |  coverage {result.coverage:.0%}",
            use_container_width=True,
        )
    else:
        st.image(decode_for_display(image_bytes),
                 caption="Eredeti kép (ROI nem detektálható)",
                 use_container_width=True)

    # Top-3 sávdiagram CNN esetén
    if use_cnn and len(result.top3) > 1:
        st.divider()
        st.subheader("Top-3 valószínűség")
        for cls_name, prob in result.top3:
            col_lbl, col_bar = st.columns([1, 5])
            col_lbl.write(f"**{cls_name}**")
            col_bar.progress(float(prob), text=f"{prob:.1%}")

# ─── DIAGNOSTIC MODE ON → 16-panel audit ─────────────────────────────────────
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.viz_diagnostics import create_full_pipeline_audit

    st.divider()
    st.subheader("Pipeline Audit – 16 panel")

    image_bgr = preprocess_image_input(image_bytes, max_long_edge=_UPLOAD_MAX_PX)

    with st.spinner("Audit vizualizáció generálása..."):
        fig = create_full_pipeline_audit(image_bgr, result.pipeline_result)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
