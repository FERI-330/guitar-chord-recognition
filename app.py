"""
app.py – Streamlit demo: Gitár Akkord Felismerő

UI-logika. Minden pipeline-logika src/inference.py-ban van, hogy az
eredmények 100%-ban azonosak legyenek a notebook run_v14_pipeline
kimenetével.

Módok:
  Diagnostic OFF → canon_norm kép + akkord neve + pipeline debug expander
  Diagnostic ON  → 16 paneles viz_diagnostics audit nézet

Indítás:
    streamlit run app.py
"""
import sys
from pathlib import Path
from typing import Any

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

# Large arrays stripped from the JSON debug view to keep it readable.
_STRIP_KEYS = frozenset({
    "img", "canon", "canon_norm", "canon_pre_shear",
    "finger_mask", "hand_mask", "H", "H_inv",
    "edges", "edges_masked",
})


def _to_json(value: Any) -> Any:
    """Recursively convert pipeline result values to JSON-serializable types.

    Large numpy arrays are replaced with a shape descriptor string.
    """
    if isinstance(value, np.ndarray):
        if value.size <= 20:
            return value.tolist()
        return f"<ndarray shape={list(value.shape)} dtype={value.dtype}>"
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items() if k not in _STRIP_KEYS}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _pipeline_debug_values(pr: dict) -> dict:
    """Extract key scalar diagnostics from the raw pipeline result dict."""
    shear = pr.get("shear") or {}
    nut = pr.get("nut") or {}
    fit = pr.get("fit") or {}

    # Neck span from trapezoid corner positions (TL → TR edge length)
    corners = (pr.get("trap") or {}).get("corners_px")
    span_px = None
    if corners is not None:
        c = np.asarray(corners, dtype=np.float64)
        span_px = float(np.linalg.norm(c[1] - c[0]))

    return {
        "nut_x": nut.get("nut_x"),
        "shear_angle_deg": shear.get("shear_angle_deg"),
        "shear_corrected": shear.get("corrected", False),
        "shear_n_lines": shear.get("n_lines", 0),
        "span_px": span_px,
        "warp_stretch": pr.get("warp_stretch_factor"),
        "coverage": (fit.get("coverage_ratio") or 0.0),
        "n_visible": fit.get("n_visible", 0),
        "is_flipped": pr.get("is_flipped"),
        "nut_direction": pr.get("nut_direction"),
        "fret_detector": pr.get("fret_detector_method"),
        "invalid_reason": pr.get("invalid_reason"),
    }


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
    st.caption("**Pipeline:** V14.1 (Hough + homográfia + 17.817 szabály)")
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

# Sub-pixel alignment: decode the preprocessed shape for consistency audit
_input_bgr = preprocess_image_input(image_bytes, max_long_edge=_UPLOAD_MAX_PX)
_input_shape = _input_bgr.shape  # (H, W, 3) after preprocess

# ─── Eredmény metrikák (mindig látható) ───────────────────────────────────────
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Akkord", result.chord)
m2.metric("Confidence", f"{result.confidence:.1%}")
m3.metric("Pipeline", "OK ✓" if result.ok else "FAIL ✗")

# ─── Pipeline Debug Expander (mindig látható) ─────────────────────────────────
with st.expander("🔍 Részletes Pipeline Diagnosztika"):
    dbg = _pipeline_debug_values(result.pipeline_result)

    # Sub-pixel alignment info
    st.caption(
        f"**Bemeneti felbontás (Hough input):** "
        f"{_input_shape[1]}×{_input_shape[0]} px  "
        f"(max_long_edge={_UPLOAD_MAX_PX}) — "
        f"notebook futásoknál max_long_edge=0 (natív felbontás)"
    )

    dc1, dc2, dc3, dc4 = st.columns(4)
    _nut_str = f"{dbg['nut_x']:.1f}" if dbg["nut_x"] is not None else "N/A"
    _shear_str = f"{dbg['shear_angle_deg']:.2f}°" if dbg["shear_angle_deg"] is not None else "N/A"
    _span_str = f"{dbg['span_px']:.0f}" if dbg["span_px"] is not None else "N/A"
    _stretch_str = f"{dbg['warp_stretch']:.2f}×" if dbg["warp_stretch"] is not None else "N/A"
    dc1.metric("Nut X (px)", _nut_str)
    dc2.metric("Shear (°)", _shear_str,
               help="step6d shear-korrekció szöge. Streamlit vs. notebook eltérés itt látszik.")
    dc3.metric("Span (px)", _span_str,
               help="Nyakszélesség (TL→TR trapézoid él) pixelben. Felbontás-függő.")
    dc4.metric("Warp stretch", _stretch_str)

    dc5, dc6, dc7, dc8 = st.columns(4)
    dc5.metric("Coverage", f"{dbg['coverage']:.0%}")
    dc6.metric("N frets matched", str(dbg["n_visible"]))
    dc7.metric("Is flipped", "Igen" if dbg["is_flipped"] else "Nem")
    dc8.metric("Fret detector", dbg["fret_detector"] or "N/A")

    if dbg["invalid_reason"]:
        st.error(f"**invalid_reason:** {dbg['invalid_reason']}")

    if dbg["nut_direction"]:
        st.info(f"**Nut irány:** {dbg['nut_direction']}")

    with st.expander("Teljes pipeline result dict (nagy tömbök nélkül)"):
        st.json(_to_json(result.pipeline_result))

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
