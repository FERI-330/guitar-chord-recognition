import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy import signal

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "output"


def _safe_gray(img_bgr):
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2:
        return img_bgr.astype(np.uint8)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _draw_bbox(ax, bbox, color=(0, 1, 0), lw=1.0):
    if bbox is None:
        return
    x, y, w, h = bbox
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)


def _angle_color(angle_deg):
    # Map -90..90 to a color map (blue->red)
    a = (angle_deg + 90.0) / 180.0
    a = np.clip(a, 0.0, 1.0)
    cmap = plt.get_cmap("RdYlBu")
    return cmap(a)


def create_full_pipeline_audit(image, results, save_path=None):
    """Create a comprehensive 4x4 diagnostics figure.

    Parameters
    - image: BGR image (numpy) or None
    - results: dict returned by the pipeline (may be partially populated)

    The function is defensive: it will use whatever is available in `results`.
    """
    # Prepare inputs
    img_bgr = None
    if image is not None:
        img_bgr = image.copy()
    else:
        img_bgr = results.get("img")
        if img_bgr is None:
            img_bgr = results.get("canon")

    if img_bgr is None:
        # Nothing to draw
        fig = plt.figure(figsize=(12, 9))
        plt.text(0.5, 0.5, "No image available for diagnostics", ha="center", va="center")
        return fig

    gray = _safe_gray(img_bgr)

    # Compute CLAHE
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
    except Exception:
        clahe_img = gray

    # Masks: try to get from results, otherwise create simple hand mask from landmarks
    hand_mask = results.get("hand_mask")
    neck_mask = results.get("neck_mask")
    landmarks = results.get("landmarks")
    if hand_mask is None and landmarks is not None:
        # landmarks expected as list of (x,y) points
        try:
            pts = np.array(landmarks, dtype=np.int32)
            hand_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillConvexPoly(hand_mask, pts, 255)
        except Exception:
            hand_mask = np.zeros(gray.shape, dtype=np.uint8)
    if neck_mask is None:
        # crude neck mask: area around median horizontal band where frets likely are
        h = gray.shape[0]
        neck_mask = np.zeros_like(gray)
        y1 = int(h * 0.25)
        y2 = int(h * 0.65)
        neck_mask[y1:y2, :] = 255

    # Edge detections
    edges_canny = cv2.Canny(gray, 50, 150)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobelx_abs = np.uint8(np.clip(np.abs(sobelx) / (sobelx.max() + 1e-6) * 255, 0, 255))

    # ROI bounding box: try results.get('roi') or compute from landmarks or neck_mask
    roi = results.get("roi")
    if roi is None and landmarks is not None:
        try:
            pts = np.array(landmarks, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            roi = (x, y, w, h)
        except Exception:
            roi = None
    if roi is None:
        # fallback to full image
        roi = (0, 0, img_bgr.shape[1], img_bgr.shape[0])

    # Canonical ROI image if available (used for mask/size alignment)
    canon_img = results.get("canon")

    # Helper: normalize mask to uint8 [0,255] and map/crop/resize into canonical ROI coords
    def _prepare_mask(mask, roi_box, target_img=None):
        if mask is None:
            # return zeros shaped to target_img or roi size
            if target_img is not None:
                h, w = target_img.shape[:2]
                return np.zeros((h, w), dtype=np.uint8)
            else:
                _, _, rw, rh = roi_box[0], roi_box[1], roi_box[2], roi_box[3]
                return np.zeros((rw, rh), dtype=np.uint8)

        mask = np.array(mask)
        # Ensure mask numeric range is [0,255] uint8
        try:
            mmax = float(mask.max()) if mask.size > 0 else 0.0
        except Exception:
            mmax = 0.0
        if mask.dtype != np.uint8:
            if mmax <= 1.1:
                mask = (mask.astype(np.float32) * 255.0).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        else:
            # uint8 but may be 0/1
            if mmax <= 1:
                mask = (mask.astype(np.uint8) * 255).astype(np.uint8)

        x, y, w, h = roi_box
        # If mask matches full image size, crop to roi
        if mask.shape[:2] == gray.shape:
            mask_roi = mask[y : y + h, x : x + w]
        elif target_img is not None and mask.shape[:2] == target_img.shape[:2]:
            mask_roi = mask
        else:
            # try to resize to roi size
            try:
                mask_roi = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                mask_roi = np.zeros((h, w), dtype=np.uint8)

        # If canonical target exists and sizes differ, resize mask into canonical size
        if target_img is not None:
            th, tw = target_img.shape[:2]
            if mask_roi.shape[:2] != (th, tw):
                try:
                    mask_roi = cv2.resize(mask_roi, (tw, th), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    mask_roi = np.zeros((th, tw), dtype=np.uint8)

        return mask_roi

    # Prepare visualizable masks (in canonical ROI coords if possible)
    hand_mask_vis = _prepare_mask(hand_mask, roi, target_img=canon_img)
    neck_mask_vis = _prepare_mask(neck_mask, roi, target_img=canon_img)

    # Debug: mask stats
    try:
        print(f"Mask stats: hand min={hand_mask_vis.min()}, max={hand_mask_vis.max()}, non-zero={np.count_nonzero(hand_mask_vis)}")
        print(f"DEBUG: Hand mask sum: {np.sum(hand_mask_vis)}")
    except Exception:
        print("Mask stats: hand mask unavailable")

    # Hough lines
    lines = None
    try:
        # use sobel or canny for Hough
        lines = cv2.HoughLinesP(edges_canny, 1, np.pi / 180.0, threshold=80, minLineLength=50, maxLineGap=10)
    except Exception:
        lines = None

    # Shear info
    shear = results.get("shear") or {}

    # Intensity profile
    profile = results.get("profile")
    if profile is None:
        profile = results.get("intensity_profile")
    profile_raw = results.get("profile_raw")
    if profile_raw is None:
        profile_raw = results.get("raw_profile")
    # prefer profile from results; if missing or empty, compute from canonical ROI
    if profile is None:
        if canon_img is not None:
            g = _safe_gray(canon_img)
            profile = g.mean(axis=0)
            if profile.max() > 1e-6:
                profile = profile / float(profile.max())
            else:
                # all zeros despite canon -> keep None
                profile = None
        else:
            profile = None

    # If profile exists but is all zeros or near-zero, recompute from raw canon (unmasked) as fallback
    profile_note = None
    try:
        if profile is not None and np.max(profile) <= 1e-6:
            if canon_img is not None:
                g = _safe_gray(canon_img)
                raw_profile = g.mean(axis=0)
                if raw_profile.max() > 1e-6:
                    profile = raw_profile / float(raw_profile.max())
                    profile_note = "Raw (no mask)"
                    print(f"[viz_diagnostics] Profile empty — recomputed from raw canonical ROI")
    except Exception:
        pass

    # Debug: ROI mean intensity
    try:
        print(f"DEBUG: ROI mean intensity: {float(np.mean(gray))}")
    except Exception:
        print("DEBUG: ROI mean intensity: None")

    # Clean up profile arrays for plotting
    def _clean_profile(arr):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    profile = _clean_profile(profile)
    profile_raw = _clean_profile(profile_raw)
    if profile_raw is None and profile is not None:
        profile_raw = profile.copy()

    # Find peaks if profile exists
    peaks = []
    peak_props = {}
    if profile is not None:
        try:
            peaks, props = signal.find_peaks(profile, prominence=0.05, width=1)
            peak_props = props
        except Exception:
            peaks = []

    # Nut info
    nut = results.get("nut") or {}

    # Fingertips / touch points
    touch_points = results.get("touch_points") or []

    # Final plotting: 4x4 grid
    fig, axs = plt.subplots(4, 4, figsize=(24, 18), constrained_layout=True)
    axs = axs.reshape(-1)

    def _safe_draw(ax, draw_fn, fallback_msg="Unavailable", show_axis=False):
        try:
            ax.cla()
            draw_fn()
        except Exception as e:
            ax.cla()
            try:
                ax.text(0.5, 0.5, f"{fallback_msg}: {type(e).__name__}: {e}", ha="center", va="center")
            except Exception:
                pass
        finally:
            if not show_axis:
                ax.axis("off")

    # Row 1
    ax = axs[0]
    def _draw_raw():
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title("Raw image (+skeleton)")
        if landmarks is not None:
            pts = np.array(landmarks)
            ax.scatter(pts[:, 0], pts[:, 1], c="cyan", s=12)

    _safe_draw(ax, _draw_raw, fallback_msg="Raw image unavailable")

    ax = axs[1]
    _safe_draw(ax, lambda: (ax.imshow(gray, cmap="gray"), ax.set_title("Gray")), fallback_msg="Gray unavailable")

    ax = axs[2]
    _safe_draw(ax, lambda: (ax.imshow(clahe_img, cmap="gray"), ax.set_title("CLAHE")), fallback_msg="CLAHE unavailable")

    ax = axs[3]
    def _draw_handmask():
        if canon_img is not None and hand_mask_vis is not None:
            ax.imshow(hand_mask_vis, cmap="gray")
            if np.count_nonzero(hand_mask_vis) == 0 and landmarks is not None:
                ax.set_title("Mask Empty (Check Warp!)")
            else:
                ax.set_title("Hand mask (canonical ROI)")
        else:
            ax.imshow(hand_mask, cmap="gray")
            ax.set_title("Hand mask")

    _safe_draw(ax, _draw_handmask, fallback_msg="Hand mask unavailable")

    # Row 2
    ax = axs[4]
    def _draw_neckmask():
        if canon_img is not None and neck_mask_vis is not None:
            ax.imshow(neck_mask_vis, cmap="gray")
            ax.set_title("Neck mask (canonical ROI)")
        else:
            ax.imshow(neck_mask, cmap="gray")
            ax.set_title("Neck mask")

    _safe_draw(ax, _draw_neckmask, fallback_msg="Neck mask unavailable")

    ax = axs[5]
    _safe_draw(ax, lambda: (ax.imshow(edges_canny, cmap="gray"), ax.set_title("Canny edges")), fallback_msg="Canny unavailable")

    # ROI on raw
    ax = axs[6]
    def _draw_roi():
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title("ROI (bbox + hull)")
        _draw_bbox(ax, roi, color=(1, 0.5, 0))
        ys, xs = np.where(neck_mask > 0)
        pts = np.vstack([xs, ys]).T
        if pts.shape[0] > 0:
            hull = cv2.convexHull(pts.astype(np.int32))
            ax.plot(hull[:, 0, 0], hull[:, 0, 1], color="yellow", linewidth=1.0)

    _safe_draw(ax, _draw_roi, fallback_msg="ROI unavailable")

    # Hough lines colored by angle
    ax = axs[7]
    def _draw_hough():
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
                ax.plot([x1, x2], [y1, y2], color=_angle_color(angle), linewidth=1.2)
        ax.set_title("Hough lines (angle color)")

    _safe_draw(ax, _draw_hough, fallback_msg="Hough unavailable")
    ax = axs[8]
    def _draw_profile():
        if profile is None:
            ax.text(0.5, 0.5, "No profile available", ha="center", va="center")
            return
        xs = np.arange(len(profile))
        ax.plot(xs, profile, color="steelblue")
        ax.fill_between(xs, profile, alpha=0.15)
        if profile_raw is not None and profile_raw is not profile:
            xs_raw = np.arange(len(profile_raw))
            ax.plot(xs_raw, profile_raw, color="steelblue", alpha=0.3, lw=1.0)
        ax.relim()
        ax.autoscale_view()
        ax.set_ylim(auto=True)
        if len(peaks) > 0:
            ax.scatter(peaks, profile[peaks], c="gray", s=20, label="peaks")
        nut_x = nut.get("nut_x") if nut else None
        candidates = results.get("nut_candidates") or []
        for c in candidates:
            ax.axvline(c.get("x", 0), color="yellow", linewidth=1.0, alpha=0.8)
        if nut_x is not None:
            ax.axvline(nut_x, color="blue", linewidth=2.0, label="selected nut")
        prom = peak_props.get("prominences") if isinstance(peak_props, dict) else None
        if prom is not None and len(prom) > 0:
            pmean = np.mean(prom)
            ax.axhline(min(1.0, pmean), color="#888", ls=":", lw=0.9, label="mean-prom")
        ax.set_title("Intensity profile & decisions")

    _safe_draw(ax, _draw_profile, fallback_msg="Profile unavailable", show_axis=True)
    # Debug: profile stats
    try:
        if profile is None:
            print("Profile stats: None")
        else:
            print(f"Profile stats: len={len(profile)}, max={np.max(profile)}")
            print(f"DEBUG: Profile max value: {np.max(profile) if profile is not None else 'None'}")
            if profile_note is not None:
                ax.text(0.02, 0.95, profile_note, transform=ax.transAxes, fontsize=8, color="#c0392b")
    except Exception:
        pass

    ax = axs[9]
    def _draw_thresholds():
        ax.axis("off")
        txt = []
        txt.append(f"find_peaks count={len(peaks)}")
        if isinstance(peak_props, dict):
            for k, v in peak_props.items():
                try:
                    txt.append(f"{k}: {np.array(v).tolist()[:5]}")
                except Exception:
                    txt.append(f"{k}: (len={len(v)})")
        ax.text(0.01, 0.98, "\n".join(txt), va="top", ha="left", fontsize=8, family="monospace")

    _safe_draw(ax, _draw_thresholds, fallback_msg="Thresholds unavailable")

    # Nut safety margin
    ax = axs[10]
    def _draw_nutmargin():
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        nm = results.get("nut_margin_px") or results.get("nut_extend_amin_margin_px") or 0
        if nut and nut.get("nut_x") is not None:
            x = int(nut.get("nut_x"))
            ax.axvline(x, color="blue", lw=2)
            ax.axvline(x - nm, color="orange", lw=1, ls="--")
            ax.axvline(x + nm, color="orange", lw=1, ls="--")
        ax.set_title("Nut safety margin & extension")

    _safe_draw(ax, _draw_nutmargin, fallback_msg="Nut margin unavailable")

    # Row 4: final mapping
    ax = axs[12]
    def _draw_canon():
        canon = results.get("canon") or img_bgr
        ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB))
        frets = results.get("fret_xs_filt") or []
        for fx in frets:
            ax.axvline(fx, color="#e74c3c", lw=1.0)
        ax.set_title("Canonical ROI + frets")

    _safe_draw(ax, _draw_canon, fallback_msg="Canonical ROI unavailable")

    ax = axs[13]
    def _draw_tips():
        canon = results.get("canon") or img_bgr
        ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB))
        for tp in touch_points:
            x, y = int(tp[0]), int(tp[1])
            ax.scatter([x], [y], c="lime", s=40)
        ax.set_title("Touch points (TIPs)")

    _safe_draw(ax, _draw_tips, fallback_msg="Touch points unavailable")

    ax = axs[14]
    def _draw_grid():
        canon = results.get("canon") or img_bgr
        ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB))
        h, w = canon.shape[:2]
        for i in range(6):
            y = int(h * (0.15 + 0.7 * i / 5.0))
            ax.axhline(y, color="#2c3e50", lw=0.6, alpha=0.6)
        ax.set_title("6-string theoretical grid")

    _safe_draw(ax, _draw_grid, fallback_msg="Grid unavailable")

    # Textual summary
    ax = axs[15]
    def _draw_text():
        ax.axis("off")
        lines = []
        lines.append(f"Status: {'OK' if results.get('ok') else 'FAIL'}")
        if results.get('intensity_profile_mode'):
            lines.append(f"Profile mode: {results.get('intensity_profile_mode')}")
        if results.get('intensity_auto_strategy'):
            lines.append(f"Auto-strategy: {results.get('intensity_auto_strategy')}")
        if shear:
            lines.append(f"Shear: {shear}")
        if not results.get('ok'):
            reason = results.get('invalid_reason', 'n/a')
            lines.append(f"Fail reason: {reason}")
        ax.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

    _safe_draw(ax, _draw_text, fallback_msg="Summary unavailable")

    if save_path is not None:
        output_path = Path(save_path)
        if not output_path.is_absolute():
            output_path = _PROJECT_ROOT / output_path
        if output_path.suffix == "":
            image_name = Path(results.get("fname", "diagnostic")).stem or "diagnostic"
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            output_path = output_path / f"diag_{timestamp}_{image_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Mentés helye: {os.path.abspath(output_path)}")

    # Fail-soft red frame
    if not results.get('ok'):
        for a in axs:
            for spine in a.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2.0)

    return fig
