import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy import signal


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
    profile = results.get("intensity_profile")
    if profile is None:
        # compute from canon if available as mean along axis 0
        canon = results.get("canon")
        if canon is not None:
            g = _safe_gray(canon)
            profile = g.mean(axis=0)
            # normalize
            if profile.max() > 1e-6:
                profile = profile / float(profile.max())
        else:
            profile = None

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

    # Row 1
    ax = axs[0]
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("Raw image (+skeleton)")
    if landmarks is not None:
        try:
            pts = np.array(landmarks)
            ax.scatter(pts[:, 0], pts[:, 1], c="cyan", s=12)
        except Exception:
            pass
    ax.axis("off")

    ax = axs[1]
    ax.imshow(gray, cmap="gray")
    ax.set_title("Gray")
    ax.axis("off")

    ax = axs[2]
    ax.imshow(clahe_img, cmap="gray")
    ax.set_title("CLAHE")
    ax.axis("off")

    ax = axs[3]
    ax.imshow(hand_mask, cmap="gray")
    ax.set_title("Hand mask")
    ax.axis("off")

    # Row 2
    ax = axs[4]
    ax.imshow(neck_mask, cmap="gray")
    ax.set_title("Neck mask")
    ax.axis("off")

    ax = axs[5]
    ax.imshow(edges_canny, cmap="gray")
    ax.set_title("Canny edges")
    ax.axis("off")

    # ROI on raw
    ax = axs[6]
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("ROI (bbox + hull)")
    _draw_bbox(ax, roi, color=(1, 0.5, 0))
    # convex hull from neck_mask
    try:
        ys, xs = np.where(neck_mask > 0)
        pts = np.vstack([xs, ys]).T
        if pts.shape[0] > 0:
            hull = cv2.convexHull(pts.astype(np.int32))
            ax.plot(hull[:, 0, 0], hull[:, 0, 1], color="yellow", linewidth=1.0)
    except Exception:
        pass
    ax.axis("off")

    # Hough lines colored by angle
    ax = axs[7]
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            ax.plot([x1, x2], [y1, y2], color=_angle_color(angle), linewidth=1.2)
    ax.set_title("Hough lines (angle color)")
    ax.axis("off")

    # Row 3: profiling
    ax = axs[8]
    if profile is not None:
        xs = np.arange(len(profile))
        ax.plot(xs, profile, color="steelblue")
        ax.fill_between(xs, profile, alpha=0.15)
        # plot peaks
        if len(peaks) > 0:
            ax.scatter(peaks, profile[peaks], c="gray", s=20, label="peaks")
        # nut candidates
        nut_x = None
        if nut:
            nut_x = nut.get("nut_x")
        # highlight candidates from results (if any)
        candidates = results.get("nut_candidates") or []
        for c in candidates:
            ax.axvline(c.get("x", 0), color="yellow", linewidth=1.0, alpha=0.8)
        if nut_x is not None:
            ax.axvline(nut_x, color="blue", linewidth=2.0, label="selected nut")
        # show prominences / widths from peak_props
        prom = peak_props.get("prominences") if isinstance(peak_props, dict) else None
        if prom is not None and len(prom) > 0:
            # draw a faint horizontal line at mean prominence normalized
            pmean = np.mean(prom)
            ax.axhline(min(1.0, pmean), color="#888", ls=":", lw=0.9, label="mean-prom")
    else:
        ax.text(0.5, 0.5, "No profile available", ha="center", va="center")
    ax.set_title("Intensity profile & decisions")

    ax = axs[9]
    # show thresholds used by find_peaks
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

    # Nut safety margin
    ax = axs[10]
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    nm = results.get("nut_margin_px") or results.get("nut_extend_amin_margin_px") or 0
    if nut and nut.get("nut_x") is not None:
        x = int(nut.get("nut_x"))
        ax.axvline(x, color="blue", lw=2)
        ax.axvline(x - nm, color="orange", lw=1, ls="--")
        ax.axvline(x + nm, color="orange", lw=1, ls="--")
    ax.set_title("Nut safety margin & extension")
    ax.axis("off")

    # Row 4: final mapping
    ax = axs[12]
    canon = results.get("canon") or img_bgr
    ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB))
    frets = results.get("fret_xs_filt") or []
    for fx in frets:
        ax.axvline(fx, color="#e74c3c", lw=1.0)
    ax.set_title("Canonical ROI + frets")
    ax.axis("off")

    ax = axs[13]
    ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB))
    # draw touch points
    for tp in touch_points:
        try:
            x, y = int(tp[0]), int(tp[1])
            ax.scatter([x], [y], c="lime", s=40)
        except Exception:
            pass
    ax.set_title("Touch points (TIPs)")
    ax.axis("off")

    ax = axs[14]
    # theoretical 6-string grid overlay
    ax.imshow(cv2.cvtColor(canon, cv2.COLOR_BGR2RGB))
    h, w = canon.shape[:2]
    for i in range(6):
        y = int(h * (0.15 + 0.7 * i / 5.0))
        ax.axhline(y, color="#2c3e50", lw=0.6, alpha=0.6)
    ax.set_title("6-string theoretical grid")
    ax.axis("off")

    # Textual summary
    ax = axs[15]
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

    if save_path is not None:
        output_path = Path(save_path)
        if output_path.suffix == "":
            image_name = Path(results.get("fname", "diagnostic")).stem or "diagnostic"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path / f"{image_name}_{timestamp}_diag.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    # Fail-soft red frame
    if not results.get('ok'):
        for a in axs:
            for spine in a.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2.0)

    return fig
