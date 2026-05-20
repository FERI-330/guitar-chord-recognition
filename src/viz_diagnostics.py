"""
src/viz_diagnostics.py

16-panel pipeline audit vizualizáció, pipeline-sorrendben.

Sor 1 – Előkészítés   : Original + Landmarks | Finger Mask | Trapézoid | Warped ROI (kézzel)
Sor 2 – Geometria (F1): Hand Mask canonical  | Pre-shear ROI | Post-shear ROI | Hough + nyak
Sor 3 – Detekció (F2) : Sobel-X canonical    | Masked Profile + Peaks | Proto Nut | Debug info
Sor 4 – Eredmény      : Final overlay         | Canonical + frets | Fingertips | Összefoglaló
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy import signal

try:
    from src.prototype_nut_detector import detect_nut_prototype as _detect_nut_proto
    from src.prototype_nut_detector import detect_inlays_prototype as _detect_inlays_proto
except Exception:
    _detect_nut_proto = None
    _detect_inlays_proto = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "output"

# ── Segédfüggvények ───────────────────────────────────────────────────────────

def _safe_gray(img_bgr):
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2:
        return img_bgr.astype(np.uint8)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _angle_color(angle_deg):
    a = np.clip((angle_deg + 90.0) / 180.0, 0.0, 1.0)
    return plt.get_cmap("RdYlBu")(a)


def _draw_bbox(ax, bbox, color=(0, 1, 0), lw=1.0):
    if bbox is None:
        return
    x, y, w, h = bbox
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)


def _overlay_mask(img_rgb, mask, color_rgb=(255, 80, 80), alpha=0.45):
    """Tint img_rgb where mask>0."""
    out = img_rgb.copy().astype(np.float32)
    tint = np.zeros_like(out)
    for c, v in enumerate(color_rgb):
        tint[:, :, c] = v
    m = (mask > 0).astype(np.float32)[:, :, np.newaxis]
    return np.clip(out * (1 - alpha * m) + tint * alpha * m, 0, 255).astype(np.uint8)


# ── Fő függvény ───────────────────────────────────────────────────────────────

def create_full_pipeline_audit(image, results, save_path=None):
    """16-panel pipeline audit a run_v14_pipeline() kimeneteből.

    Paraméterek:
        image:     BGR kép (numpy) vagy None  — ha None, results["img"]-ből jön
        results:   run_v14_pipeline() visszatérési értéke
        save_path: PNG mentési útvonal (opcionális)

    Visszaad: matplotlib Figure
    """
    # ── Bemeneti adatok előkészítése ──────────────────────────────────────────
    img_bgr = (image.copy() if image is not None
               else results.get("img") or results.get("canon"))
    if img_bgr is None:
        fig = plt.figure(figsize=(12, 9))
        plt.text(0.5, 0.5, "No image available for diagnostics",
                 ha="center", va="center", fontsize=14)
        return fig

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    landmarks   = results.get("landmarks")
    finger_mask = results.get("finger_mask")   # original-space finger mask
    hand_mask   = results.get("hand_mask")     # canonical-space hand mask (F1)
    trap        = results.get("trap")
    H_inv       = results.get("H_inv")
    canon       = results.get("canon")
    canon_pre   = results.get("canon_pre_shear")
    shear       = results.get("shear") or {}
    lines_raw   = results.get("lines") or []
    neck        = results.get("neck") or {}
    profile     = results.get("intensity_profile")
    fret_xs_raw = results.get("fret_xs_raw") or []
    fret_xs_filt = results.get("fret_xs_filt") or []
    fit         = results.get("fit") or {}
    pred_x      = fit.get("predicted_x") or {}
    fingertips  = results.get("fingertips") or []
    ok          = bool(results.get("ok", False))
    cov         = float(fit.get("coverage_ratio", 0.0))

    # Prototype nut + inlays – csak vizualizációhoz
    _proto_nut = None
    if _detect_nut_proto is not None:
        try:
            _proto_nut = _detect_nut_proto(results)
        except Exception:
            pass
    nut = _proto_nut or {}

    proto_inlays = []
    if _detect_inlays_proto is not None:
        try:
            proto_inlays = _detect_inlays_proto(results) or []
        except Exception:
            pass

    # Intenzitás profil fallback
    if profile is None and canon is not None:
        g = _safe_gray(canon)
        if g is not None:
            p = g.mean(axis=0).astype(np.float32)
            mx = float(p.max())
            profile = p / mx if mx > 1e-6 else None

    def _clean_profile(arr):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    profile = _clean_profile(profile)

    peaks, peak_props = [], {}
    if profile is not None:
        try:
            peaks, peak_props = signal.find_peaks(profile, prominence=0.05, width=1)
        except Exception:
            pass

    # Debug prints
    try:
        hm = hand_mask
        print(f"[audit] hand_mask: {hm.shape if hm is not None else None}  "
              f"non-zero={np.count_nonzero(hm) if hm is not None else 0}")
        print(f"[audit] profile: {'ok' if profile is not None else 'None'}  "
              f"peaks={len(peaks)}  frets_filt={len(fret_xs_filt)}  predicted={len(pred_x)}")
    except Exception:
        pass

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(4, 4, figsize=(26, 20), constrained_layout=True)
    axs = axs.reshape(-1)

    ok_color = "#2ecc71" if ok else "#e74c3c"
    fig.suptitle(
        f"Pipeline Audit  |  {results.get('fname', results.get('class', '?'))}  |  "
        f"{'✓ OK' if ok else '✗ ' + str(results.get('invalid_reason', ''))}  |  "
        f"cov={cov:.3f}  fitted={len(pred_x)}  raw={len(fret_xs_filt)}",
        fontsize=12, fontweight="bold", color=ok_color,
    )

    def _safe_draw(ax, draw_fn, fallback_msg="Unavailable", show_axis=False):
        try:
            ax.cla()
            draw_fn()
        except Exception as e:
            ax.cla()
            try:
                ax.text(0.5, 0.5, f"{fallback_msg}\n{type(e).__name__}: {e}",
                        ha="center", va="center", fontsize=7,
                        color="red", transform=ax.transAxes, wrap=True)
            except Exception:
                pass
        finally:
            if not show_axis:
                ax.axis("off")

    # ═══════════════════════════════════════════════════════════════════════════
    # SOR 1 – ELŐKÉSZÍTÉS
    # ═══════════════════════════════════════════════════════════════════════════

    # [0] Original kép + MediaPipe Landmarks
    ax = axs[0]
    def _draw_landmarks_on_raw():
        ax.imshow(img_rgb)
        if landmarks is not None and len(landmarks) >= 21:
            h_img, w_img = img_bgr.shape[:2]
            pts = np.array([[lx * w_img, ly * h_img] for (lx, ly, _) in landmarks],
                           dtype=np.float32)
            _CONNECTIONS = [
                (0,1),(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17),
                (1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),
                (13,14),(14,15),(15,16),(17,18),(18,19),(19,20),
            ]
            for a, b in _CONNECTIONS:
                ax.plot([pts[a,0], pts[b,0]], [pts[a,1], pts[b,1]],
                        color="#f39c12", lw=0.9, alpha=0.8)
            tip_idxs = {4, 8, 12, 16, 20}
            for idx, (px, py) in enumerate(pts):
                c = "#e74c3c" if idx in tip_idxs else "#3498db"
                ax.scatter([px], [py], c=c, s=10, zorder=5)
        n_lm = len(landmarks) if landmarks is not None else 0
        ax.set_title(f"Original + MediaPipe ({n_lm} landmarks)", fontsize=9)

    _safe_draw(ax, _draw_landmarks_on_raw, fallback_msg="Original unavailable")

    # [1] Finger Mask (original image space)
    ax = axs[1]
    def _draw_finger_mask():
        if finger_mask is not None and finger_mask.ndim == 2:
            vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
            nz = int(np.count_nonzero(finger_mask))
            total = finger_mask.size
            pct = 100.0 * nz / max(total, 1)
            vis = _overlay_mask(vis, finger_mask, color_rgb=(220, 60, 220))
            ax.imshow(vis)
            ax.set_title(f"Finger Mask (original)  {nz}px / {pct:.1f}%", fontsize=9)
        else:
            ax.imshow(img_rgb)
            ax.text(0.5, 0.5, "Finger mask\nn/a", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=10)
            ax.set_title("Finger Mask (original)", fontsize=9)

    _safe_draw(ax, _draw_finger_mask, fallback_msg="Finger mask unavailable")

    # [2] Trapézoid
    ax = axs[2]
    def _draw_trap():
        ax.imshow(img_rgb)
        corners = None
        if trap is not None and "corners_px" in trap:
            corners = np.asarray(trap["corners_px"], dtype=np.int32).reshape(4, 2)
        if corners is not None:
            poly = plt.Polygon(corners, fill=False, edgecolor="#2ecc71", linewidth=2)
            ax.add_patch(poly)
            ax.scatter(corners[:, 0], corners[:, 1],
                       c=["#e74c3c", "#f39c12", "#3498db", "#9b59b6"], s=40, zorder=5)
        trap_ok = bool(results.get("trap_ok", False))
        reasons = results.get("trap_reasons") or []
        ttl = "Trapézoid: ✓" if trap_ok else "Trapézoid: ✗"
        if reasons:
            ttl += f"  [{reasons[0]}]"
        ax.set_title(ttl, fontsize=9,
                     color="#2ecc71" if trap_ok else "#e74c3c")

    _safe_draw(ax, _draw_trap, fallback_msg="Trapezoid unavailable")

    # [3] Warped ROI — hand visible (F1: raw image warped, not masked)
    ax = axs[3]
    def _draw_warped_roi():
        if canon is not None:
            ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB), aspect="auto")
            ax.set_title(f"Warped ROI (kézzel, F1)  {canon.shape[1]}×{canon.shape[0]}px",
                         fontsize=9)
        else:
            ax.set_facecolor("#f8e8e8")
            reason = results.get("invalid_reason", "n/a")
            ax.text(0.5, 0.5, f"Canon n/a\n{reason}", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=9)
            ax.set_title("Warped ROI (n/a)", fontsize=9)

    _safe_draw(ax, _draw_warped_roi, fallback_msg="Warped ROI unavailable")

    # ═══════════════════════════════════════════════════════════════════════════
    # SOR 2 – GEOMETRIA (F1)
    # ═══════════════════════════════════════════════════════════════════════════

    # [4] Hand Mask in Canonical Space
    ax = axs[4]
    def _draw_hand_mask_canon():
        if hand_mask is not None and hand_mask.ndim == 2:
            nz = int(np.count_nonzero(hand_mask))
            pct = 100.0 * nz / max(hand_mask.size, 1)
            if nz == 0:
                if canon is not None:
                    ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB), aspect="auto",
                              alpha=0.4)
                ax.text(0.5, 0.5, "EMPTY MASK\n(warp error?)", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=11,
                        fontweight="bold")
                ax.set_title("Hand Mask (canonical)  ← EMPTY", fontsize=9, color="red")
            else:
                # Overlay mask on canonical ROI
                if canon is not None:
                    vis = cv2.cvtColor(canon, cv2.COLOR_BGR2RGB)
                    vis = _overlay_mask(vis, hand_mask, color_rgb=(180, 60, 220))
                    ax.imshow(vis, aspect="auto")
                else:
                    ax.imshow(hand_mask, cmap="magma", aspect="auto")
                ax.set_title(f"Hand Mask (canonical)  {nz}px / {pct:.1f}%", fontsize=9)
        else:
            if canon is not None:
                ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB), aspect="auto")
            ax.text(0.5, 0.5, "Hand mask\nn/a", ha="center", va="center",
                    transform=ax.transAxes, color="orange", fontsize=9)
            ax.set_title("Hand Mask (canonical)  n/a", fontsize=9)

    _safe_draw(ax, _draw_hand_mask_canon, fallback_msg="Hand mask unavailable")

    # [5] Pre-shear Canonical ROI
    ax = axs[5]
    def _draw_pre_shear():
        src = canon_pre if canon_pre is not None else canon
        if src is not None:
            ax.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), aspect="auto")
            lbl = "Pre-shear ROI" if canon_pre is not None else "Canon (shear n/a)"
            ax.set_title(lbl, fontsize=9)
        else:
            ax.set_facecolor("#f0f0f0")
            ax.set_title("Pre-shear ROI  n/a", fontsize=9)

    _safe_draw(ax, _draw_pre_shear, fallback_msg="Pre-shear unavailable")

    # [6] Shear-corrected Canonical ROI
    ax = axs[6]
    def _draw_post_shear():
        corrected = bool(shear.get("corrected", False))
        shear_deg = float(shear.get("shear_angle_deg", 0.0))
        residual  = float(shear.get("residual_shear_deg", 0.0))
        conf      = float(shear.get("hough_confidence", 0.0))
        if corrected and canon is not None:
            ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB), aspect="auto")
            ax.set_title(
                f"Shear-korrigált ROI  Δ={shear_deg:.1f}°  res={residual:.2f}°",
                fontsize=9, color="#2ecc71",
            )
        elif canon is not None:
            ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB), aspect="auto")
            ax.set_title(
                f"Canon (nem korrigált)  shear={shear_deg:.1f}°  conf={conf:.2f}",
                fontsize=9, color="#888888",
            )
        else:
            ax.set_facecolor("#f0f0f0")
            ax.set_title("Shear info  n/a", fontsize=9)

    _safe_draw(ax, _draw_post_shear, fallback_msg="Shear unavailable")

    # [7] Hough Lines + Neck Angle
    ax = axs[7]
    def _draw_hough():
        ax.imshow(img_rgb)
        if lines_raw:
            for ln in lines_raw:
                x1, y1, x2, y2 = ln
                ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                ax.plot([x1, x2], [y1, y2], color=_angle_color(ang), lw=1.0, alpha=0.7)
        ang_deg = float(neck.get("angle_deg", 0.0))
        n_long  = len((results.get("split") or {}).get("long_lines") or [])
        n_fret  = len((results.get("split") or {}).get("fret_lines") or [])
        ax.set_title(
            f"Hough + Nyak  angle={ang_deg:.1f}°  "
            f"long={n_long}  fret_lines={n_fret}",
            fontsize=9,
        )

    _safe_draw(ax, _draw_hough, fallback_msg="Hough unavailable")

    # ═══════════════════════════════════════════════════════════════════════════
    # SOR 3 – DETEKCIÓ (F2)
    # ═══════════════════════════════════════════════════════════════════════════

    # [8] Sobel-X gradient on canonical
    ax = axs[8]
    def _draw_sobel_canon():
        src = canon if canon is not None else img_bgr
        gray_c = _safe_gray(src)
        if gray_c is None:
            ax.set_title("Sobel-X  n/a", fontsize=9)
            return
        sx = cv2.Sobel(gray_c, cv2.CV_32F, 1, 0, ksize=3)
        sx_abs = np.abs(sx)
        mx = float(sx_abs.max())
        if mx > 0:
            sx_abs = sx_abs / mx
        ax.imshow(sx_abs, cmap="inferno", aspect="auto",
                  vmin=0.0, vmax=1.0)
        ax.set_title("Sobel-X Gradient (canonical)", fontsize=9)

    _safe_draw(ax, _draw_sobel_canon, fallback_msg="Sobel unavailable")

    # [9] Masked Intensity Profile + Peaks
    ax = axs[9]
    def _draw_masked_profile():
        if profile is None:
            ax.text(0.5, 0.5, "No profile", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title("Masked Profile  n/a", fontsize=9)
            return
        xs = np.arange(len(profile))
        ax.fill_between(xs, profile, alpha=0.18, color="steelblue")
        ax.plot(xs, profile, color="steelblue", lw=1.2)
        # Raw detected peaks
        if len(peaks) > 0:
            ax.scatter(peaks, profile[peaks], c="#e74c3c", s=25, zorder=5,
                       label=f"peaks ({len(peaks)})")
        # Raw fret detections
        for i, fx in enumerate(fret_xs_raw):
            ax.axvline(fx, color="#e67e22", lw=0.7, alpha=0.6,
                       label="raw" if i == 0 else "")
        # Fitted fret positions (green)
        for i, (_, pfx) in enumerate(pred_x.items()):
            ax.axvline(pfx, color="#2ecc71", lw=0.9, alpha=0.8,
                       label="fitted" if i == 0 else "")
        # Proto nut
        nut_x = nut.get("nut_x") if nut else None
        if nut_x is not None:
            ax.axvline(nut_x, color="yellow", lw=1.5, ls="--",
                       label=f"nut @{int(nut_x)}px")
        ax.set_xlim(0, len(profile))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
        ax.grid(alpha=0.2)
        mode = results.get("intensity_profile_mode", "?")
        ax.set_title(
            f"Profile ({mode})  peaks={len(peaks)}  "
            f"raw={len(fret_xs_raw)}  fitted={len(pred_x)}",
            fontsize=9,
        )

    _safe_draw(ax, _draw_masked_profile, fallback_msg="Profile unavailable",
               show_axis=True)

    # [10] Prototype Nut Visualization
    ax = axs[10]
    def _draw_proto_nut():
        src = canon if canon is not None else img_bgr
        h_src = src.shape[0]
        ax.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), aspect="auto")
        nut_x = nut.get("nut_x") if nut else None
        nut_side = nut.get("side", "?") if nut else "?"
        if nut_x is not None:
            # Translucent yellow band ±4 px around nut position
            ax.axvspan(nut_x - 4, nut_x + 4, color="yellow", alpha=0.35, zorder=2)
            ax.axvline(nut_x, color="yellow", lw=1.5, ls="--", zorder=3,
                       label=f"nut @{int(nut_x)}px side={nut_side}")
        # Inlay blue dots (prototype)
        for inl in proto_inlays:
            cx = inl.get("canon_x")
            if cx is not None:
                ax.scatter([cx], [h_src / 2], c="#3498db", s=28, zorder=5,
                           marker="o", alpha=0.85)
        if proto_inlays:
            ax.scatter([], [], c="#3498db", s=28, marker="o",
                       label=f"inlays ({len(proto_inlays)})")
        if nut_x is not None or proto_inlays:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.75)
        title_parts = [f"Proto Nut+Inlay (debug only)"]
        if nut_x is not None:
            title_parts.append(f"nut={int(nut_x)}px {nut_side}"
                                f"{' ⚠' if nut.get('safety') else ''}")
        title_parts.append(f"inlays={len(proto_inlays)}")
        ax.set_title("  ".join(title_parts), fontsize=8)

    _safe_draw(ax, _draw_proto_nut, fallback_msg="Proto nut unavailable")

    # [11] Debug / Detection Info
    ax = axs[11]
    def _draw_debug_info():
        ax.axis("off")
        lines_txt = []
        lines_txt.append(f"── Fit ──────────────────────")
        lines_txt.append(f"method: {fit.get('fit_method', '?')}")
        lines_txt.append(f"coverage: {cov:.3f}")
        lines_txt.append(f"inliers: {fit.get('inlier_count', '?')}/{fit.get('n_visible', '?')}")
        lines_txt.append(f"avg_res: {fit.get('avg_residual_px', 0.0):.2f}px")
        lines_txt.append(f"offset: {fit.get('offset', 0.0):.1f}  scale: {fit.get('scale', 0.0):.1f}")
        lines_txt.append(f"── Peaks ────────────────────")
        lines_txt.append(f"count: {len(peaks)}")
        if isinstance(peak_props, dict):
            prom = peak_props.get("prominences")
            wid  = peak_props.get("widths")
            if prom is not None and len(prom) > 0:
                lines_txt.append(f"prom:  {[round(float(v), 2) for v in prom[:5]]}")
            if wid is not None and len(wid) > 0:
                lines_txt.append(f"width: {[round(float(v), 1) for v in wid[:5]]}")
        lines_txt.append(f"── Shear ────────────────────")
        lines_txt.append(f"corrected: {shear.get('corrected', False)}")
        lines_txt.append(f"angle: {shear.get('shear_angle_deg', 0.0):.2f}°")
        lines_txt.append(f"conf: {shear.get('hough_confidence', 0.0):.3f}")
        lines_txt.append(f"── Detektor ─────────────────")
        lines_txt.append(f"method: {results.get('fret_detector_method', '?')}")
        lines_txt.append(f"mode: {results.get('intensity_profile_mode', '?')}")
        auto = results.get("intensity_auto_strategy") or {}
        if auto:
            lines_txt.append(f"auto: {auto.get('reason', '?')}")
        ax.text(0.02, 0.98, "\n".join(lines_txt), va="top", ha="left",
                fontsize=7.5, family="monospace", transform=ax.transAxes)
        ax.set_title("Detection Debug Info", fontsize=9)

    _safe_draw(ax, _draw_debug_info, fallback_msg="Debug info unavailable")

    # ═══════════════════════════════════════════════════════════════════════════
    # SOR 4 – EREDMÉNY
    # ═══════════════════════════════════════════════════════════════════════════

    # [12] Final Overlay on Original (back-projected frets via H_inv)
    ax = axs[12]
    def _draw_final_overlay():
        if img_bgr is None:
            ax.set_facecolor("#f0f0f0")
            ax.set_title("Final overlay  n/a", fontsize=9)
            return
        try:
            from src.viz import PipelineVisualizer
            viz = PipelineVisualizer()
            overlay = viz.draw_fretboard_overlay(img_bgr, results)
            if landmarks is not None:
                overlay = viz.draw_landmarks(overlay, landmarks)
            ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        except Exception:
            ax.imshow(img_rgb)
        n_pred = len(pred_x)
        nut_side = results.get("nut_side_hint", "?")
        ax.set_title(
            f"Final Overlay  frets={n_pred}  side={nut_side}",
            fontsize=9, color=ok_color,
        )

    _safe_draw(ax, _draw_final_overlay, fallback_msg="Final overlay unavailable")

    # [13] Canonical ROI + Fitted Frets
    ax = axs[13]
    def _draw_canon_frets():
        src = canon if canon is not None else img_bgr
        h_src = src.shape[0]
        ax.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), aspect="auto")
        # Raw filtered – dim gray
        for fx in fret_xs_filt:
            ax.axvline(fx, color="#888888", lw=0.6, alpha=0.5)
        # Fitted – green
        for _, pfx in pred_x.items():
            ax.axvline(pfx, color="#2ecc71", lw=1.0)
        # Proto nut – yellow band
        nut_x = nut.get("nut_x") if nut else None
        if nut_x is not None:
            ax.axvspan(nut_x - 4, nut_x + 4, color="yellow", alpha=0.30, zorder=2)
            ax.axvline(nut_x, color="yellow", lw=1.2, ls="--", zorder=3,
                       label="nut (proto)")
        # Inlay blue dots
        for inl in proto_inlays:
            cx = inl.get("canon_x")
            if cx is not None:
                ax.scatter([cx], [h_src / 2], c="#3498db", s=20, zorder=5,
                           marker="o", alpha=0.85)
        n_fitted = len(pred_x)
        ax.set_title(f"Canonical ROI + frets ({n_fitted} fitted)  inlays={len(proto_inlays)}",
                     fontsize=9)

    _safe_draw(ax, _draw_canon_frets, fallback_msg="Canonical ROI unavailable")

    # [14] Fingertips in Canonical Space
    ax = axs[14]
    def _draw_fingertips():
        src = canon if canon is not None else img_bgr
        ax.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB), aspect="auto")
        _FINGER_COLORS = {4: "#e74c3c", 8: "#f39c12", 12: "#2ecc71",
                          16: "#3498db", 20: "#9b59b6"}
        _FINGER_NAMES  = {4: "Th", 8: "Idx", 12: "Mid", 16: "Rng", 20: "Pnk"}
        for ft in fingertips:
            cx = ft.get("canon_x")
            cy = ft.get("canon_y")
            tip = ft.get("tip_idx")
            if cx is None or cy is None:
                continue
            c = _FINGER_COLORS.get(tip, "white")
            n = _FINGER_NAMES.get(tip, "?")
            fret = ft.get("fret_est")
            lbl = f"{n}" if fret is None else f"{n}@f{fret:.0f}"
            ax.scatter([cx], [cy], c=c, s=60, zorder=5)
            ax.annotate(lbl, (cx, cy), textcoords="offset points",
                        xytext=(3, -8), fontsize=6, color=c)
        ax.set_title(f"Fingertips in Canon  ({len(fingertips)} tips)", fontsize=9)

    _safe_draw(ax, _draw_fingertips, fallback_msg="Fingertips unavailable")

    # [15] Pipeline Summary
    ax = axs[15]
    def _draw_summary():
        ax.axis("off")
        lines_txt = []
        lines_txt.append(f"── Pipeline Összefoglaló ────")
        lines_txt.append(f"Státusz : {'✓ OK' if ok else '✗ FAIL'}")
        if not ok:
            lines_txt.append(f"Hiba    : {results.get('invalid_reason', 'n/a')}")
        lines_txt.append(f"Osztály : {results.get('class', '?')}")
        lines_txt.append(f"Fájl    : {results.get('fname', results.get('path', '?'))}")
        lines_txt.append(f"")
        lines_txt.append(f"── Bund detektálás ──────────")
        lines_txt.append(f"Raw detek  : {len(fret_xs_raw)}")
        lines_txt.append(f"Filt detek : {len(fret_xs_filt)}")
        lines_txt.append(f"Fitted     : {len(pred_x)}")
        lines_txt.append(f"Coverage   : {cov:.4f}")
        lines_txt.append(f"")
        lines_txt.append(f"── Fitted X pozíciók ────────")
        for fn, fx in sorted(pred_x.items()):
            lines_txt.append(f"  Fret {int(fn):2d} → {float(fx):.1f}px")
        lines_txt.append(f"")
        lines_txt.append(f"── Prototype Nut ────────────")
        if nut:
            lines_txt.append(f"  x={nut.get('nut_x', 'n/a')}  side={nut.get('side', '?')}")
            lines_txt.append(f"  safety={nut.get('safety', False)}")
        else:
            lines_txt.append("  n/a")
        lines_txt.append(f"── Prototype Inlays ─────────")
        if proto_inlays:
            for inl in proto_inlays[:6]:
                cx = inl.get("canon_x")
                cf = inl.get("confidence", 0.0)
                lines_txt.append(f"  x={cx:.1f}  conf={cf:.2f}")
            if len(proto_inlays) > 6:
                lines_txt.append(f"  … +{len(proto_inlays)-6} more")
        else:
            lines_txt.append("  n/a")
        ax.text(0.02, 0.98, "\n".join(lines_txt), va="top", ha="left",
                fontsize=7.5, family="monospace", transform=ax.transAxes,
                color=ok_color if not ok else "black")
        ax.set_title("Pipeline Summary", fontsize=9)

    _safe_draw(ax, _draw_summary, fallback_msg="Summary unavailable")

    # ── Vörös keret FAIL esetén ───────────────────────────────────────────────
    if not ok:
        for a in axs:
            for spine in a.spines.values():
                spine.set_edgecolor("#e74c3c")
                spine.set_linewidth(1.5)

    # ── Mentés ────────────────────────────────────────────────────────────────
    if save_path is not None:
        output_path = Path(save_path)
        if not output_path.is_absolute():
            output_path = _PROJECT_ROOT / output_path
        if output_path.suffix == "":
            stem = Path(results.get("fname", "diagnostic")).stem or "diagnostic"
            ts = datetime.now().strftime("%y%m%d_%H%M%S")
            output_path = output_path / f"diag_{ts}_{stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Mentés helye: {os.path.abspath(output_path)}")

    return fig
