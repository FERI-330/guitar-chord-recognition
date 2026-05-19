"""
src/viz.py

Pipeline és training eredmények vizualizálása.

Publikus API:
  PipelineVisualizer          – OOP vizualizátor osztály (pipeline overlay + köztes fázisok)
  draw_pipeline_result()      – egyszerű kétpaneles segédfüggvény (backward compat)
  draw_pipeline_grid()        – batch diagnosztika grid
  plot_training_history()     – Phase-A/B görbék
  plot_multi_training_histories() – modellek összehasonlítása
  plot_scatter_2d()           – PCA/t-SNE scatter

Tervezési elvek:
  - PipelineVisualizer nem definiál globális állapotot; minden paramétert
    a konstruktor vagy az adott metódus kap meg.
  - A matematikai logika kizárólag a geometry.py / fretboard.py / constants.py
    hívásain keresztül jön létre – ez a modul csak megjelenítéssel foglalkozik.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

plt.rcParams["figure.dpi"] = 96
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["image.interpolation"] = "bilinear"

# Ensure the Agg (non-interactive) backend doesn't silently swallow plt.show() calls.
# In Jupyter notebooks %matplotlib inline overrides this; in scripts with a display
# plt.ion() would switch to interactive mode.  We detect the backend at import time
# and switch only when the caller hasn't already set up a GUI/inline backend.
import matplotlib as _mpl
if _mpl.get_backend().lower() in ("agg", ""):
    try:
        _mpl.use("module://matplotlib_inline.backend_inline")
    except Exception:
        pass  # not in a notebook environment – leave Agg as-is

from src.config import CFG, PATHS, VIS_LINE_THICKNESS
from src.constants import (
    CANONICAL_W, CANONICAL_H,
    FINGER_TIP_IDX, FINGER_CHAINS,
)

MAX_FIG_WIDTH = 12.0
MAX_FIG_HEIGHT = 6.0


# ─────────────────────────────────────────────────────────────────────────────
# Modul-szintű segédletek (a standalone függvényekhez, NEM a class-ban)
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#aaffc3",
]


def class_color(idx: int) -> tuple:
    """Index → BGR tuple (OpenCV)."""
    rgb = tuple(int(255 * c) for c in to_rgb(_CLASS_COLORS[idx % len(_CLASS_COLORS)]))
    return (rgb[2], rgb[1], rgb[0])


def class_color_mpl(idx: int) -> str:
    """Index → matplotlib hex string."""
    return _CLASS_COLORS[idx % len(_CLASS_COLORS)]


def _save_figure(fig: plt.Figure, save_path, dpi: int = 120) -> None:
    """Format-aware save: JPEG paths use quality=85, PNG otherwise."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".jpg", ".jpeg"):
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0,
                    pil_kwargs={"quality": 85, "optimize": True})
    else:
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0)


# ─────────────────────────────────────────────────────────────────────────────
# PipelineVisualizer – OOP vizualizátor
# ─────────────────────────────────────────────────────────────────────────────

class PipelineVisualizer:
    """V14 pipeline eredmények OpenCV alapú vizualizátora.

    Minden megjelenítési paraméter (szín, vastagság, betűméret) a konstruktorban
    adható meg – a példány után egyetlen globális változót sem olvas.

    A matematikai logika a meglévő src modulokból érkezik:
      - src.geometry.step1_canny / step2_hough  →  get_intermediate_plots()
      - result['H_inv'] + result['fit']['predicted_x']  →  draw_fretboard_overlay()
      - src.constants.FINGER_CHAINS / FINGER_TIP_IDX   →  draw_landmarks()

    Használat::

        viz = PipelineVisualizer()
        overlay = viz.draw_fretboard_overlay(img, pipeline_result)
        lm_img  = viz.draw_landmarks(img, landmarks)
        phases  = viz.get_intermediate_plots(img, finger_mask)
    """

    # Standard MediaPipe kézcsontváz kapcsolatok (modul-szintű konstans
    # az osztályon belül, de NEM példányszintű állapot).
    _HAND_CONNECTIONS: list[tuple[int, int]] = [
        # Csukló → MCP
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # Tenyérkeresztek
        (5, 9), (9, 13), (13, 17),
        # Hüvelykujj lánc
        (1, 2), (2, 3), (3, 4),
        # Mutató lánc
        (5, 6), (6, 7), (7, 8),
        # Középső lánc
        (9, 10), (10, 11), (11, 12),
        # Gyűrűs lánc
        (13, 14), (14, 15), (15, 16),
        # Kisujj lánc
        (17, 18), (18, 19), (19, 20),
    ]

    def _resolve_line_thickness(self, image: Optional[np.ndarray] = None) -> int:
        """Return a display thickness that scales with image width."""
        base = max(1, int(self.line_thickness))
        if image is None:
            return base
        width = int(image.shape[1])
        return max(base, max(2, int(width / 200)))

    def _draw_outlined_line(
        self,
        image: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple,
        thickness: int,
        outline_thickness: Optional[int] = None,
    ) -> None:
        """Draw a high-contrast line with a thin black outline."""
        outline = outline_thickness if outline_thickness is not None else thickness + 2
        cv2.line(image, pt1, pt2, (0, 0, 0), outline, lineType=cv2.LINE_AA)
        cv2.line(image, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    def _figure_size_for_image(self, image: np.ndarray, cols: int = 1, rows: int = 1) -> tuple[float, float]:
        """Scale figure size with the input image dimensions."""
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return (5.5 * cols, 3.5 * rows)
        width = max(5.5, (w / 100.0) * cols)
        height_ratio = h / w
        height = width * height_ratio * (rows / cols)
        return (width, height)

    def _figure_size_for_canonical(self, cols: int = 1, rows: int = 1) -> tuple[float, float]:
        """Scale canonical-space figures to stay readable on HiDPI screens."""
        width = max(5.5, (CANONICAL_W / 110.0) * cols)
        height_ratio = CANONICAL_H / CANONICAL_W
        height = width * height_ratio * (rows / cols)
        return (width, height)

    def _resolve_figsize_with_scale(
        self,
        requested: Optional[tuple],
        fallback: tuple[float, float],
    ) -> tuple[tuple[float, float], float]:
        """Resolve figure size while preserving aspect ratio under a width cap."""
        if requested is None:
            width, height = fallback
        else:
            req_w, req_h = float(requested[0]), float(requested[1])
            width = max(req_w, fallback[0])
            height = max(req_h, fallback[1])
        scale = 1.0
        if width > MAX_FIG_WIDTH:
            scale = MAX_FIG_WIDTH / width
            width *= scale
            height *= scale
        if height > MAX_FIG_HEIGHT:
            h_scale = MAX_FIG_HEIGHT / height
            scale *= h_scale
            width *= h_scale
            height *= h_scale
        return (width, height), scale

    def _resolve_figsize(self, requested: Optional[tuple], fallback: tuple[float, float]) -> tuple[float, float]:
        """Backward-compatible figure size resolver."""
        fig_size, _ = self._resolve_figsize_with_scale(requested, fallback)
        return fig_size

    def _font_size(self, base: float, scale: float, minimum: float = 7.0) -> float:
        """Scale display fonts down when the figure is width-capped."""
        return max(minimum, base * scale)

    def _line_width(self, base: float, scale: float, minimum: float = 1.0) -> float:
        """Scale display line widths down when the figure is width-capped."""
        return max(minimum, base * scale)

    def _enable_constrained_layout(self, fig: plt.Figure) -> None:
        """Prefer notebook-friendly layout management for interactive display."""
        fig.set_constrained_layout(True)

    def _set_equal_aspect(self, ax: plt.Axes) -> None:
        """Keep image pixels square unless a plot explicitly needs a different aspect."""
        ax.set_aspect("equal", adjustable="box")

    def __init__(
        self,
        neck_color: tuple = (50, 220, 50),
        fret_color: tuple = (80, 80, 240),
        landmark_color: tuple = (0, 230, 230),
        connection_color: tuple = (230, 160, 0),
        fingertip_color: tuple = (0, 180, 255),
        line_thickness: int = CFG["vis_line_thickness"],
        point_radius: int = 5,
        font_scale: float = 0.50,
        hough_line_color: tuple = (0, 200, 100),
    ) -> None:
        """Inicializálás.

        Args:
            neck_color:        BGR – gitárnyak trapéz körvonal
            fret_color:        BGR – visszavetített bund vonalak
            landmark_color:    BGR – MediaPipe landmark pontok (nem ujjhegy)
            connection_color:  BGR – ujj-csontváz vonalak
            fingertip_color:   BGR – ujjhegy kiemelő szín
            line_thickness:    alapértelmezett vonalvastagság (px)
            point_radius:      landmark kör sugara (px)
            font_scale:        OpenCV betűméret-szorzó
            hough_line_color:  BGR – Hough-detektált vonalak a köztes ábrákon
        """
        self.neck_color = neck_color
        self.fret_color = fret_color
        self.landmark_color = landmark_color
        self.connection_color = connection_color
        self.fingertip_color = fingertip_color
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.font_scale = font_scale
        self.hough_line_color = hough_line_color

    # ──────────────────────────────────────────────────────────────────────
    # draw_fretboard_overlay
    # ──────────────────────────────────────────────────────────────────────

    def draw_fretboard_overlay(
        self,
        image: np.ndarray,
        points: dict,
        direction: Optional[str] = None,
    ) -> np.ndarray:
        """Fogólap körvonal és bund vonalak berajzolása az eredeti képre.

        A V14 homográfia inverze (``H_inv``) segítségével a kanonikus térből
        (600×80 px) visszavetíti a 17.817-es szabályból kapott bund-pozíciókat
        az eredeti kép koordináta-rendszerébe.  A trapéz sarokpontok már
        képpixel-koordinátákban érkeznek (``corners_trim`` / ``trap.corners_px``),
        ezért azok matematikai újraszámítás nélkül rajzolhatók ki.

        Args:
            image:     BGR numpy ndarray (nem módosítja az eredetit).
            points:    ``run_v14_pipeline`` result dict.
                       Szükséges kulcsok (ha ``ok=True``):
                       ``'H_inv'``, ``'fit'``, ``'trap'`` vagy ``'corners_trim'``.
            direction: Nut oldal (``'left'`` / ``'right'``); ha ``None``,
                       a ``points['nut_side_hint']`` értékét veszi át.

        Returns:
            BGR numpy ndarray – overlay-es kép.
        """
        vis = image.copy()
        if not isinstance(points, dict):
            return vis

        # ── 1. Gitárnyak trapéz körvonal ──────────────────────────────────
        corners_trim = points.get("corners_trim")
        trap = points.get("trap")
        raw_corners = (
            corners_trim if corners_trim is not None
            else (trap["corners_px"] if trap is not None else None)
        )
        if raw_corners is not None:
            corners = np.asarray(raw_corners, dtype=np.int32).reshape(4, 2)
            cv2.polylines(
                vis, [corners.reshape(-1, 1, 2)], isClosed=True,
                color=self.neck_color, thickness=self.line_thickness,
            )
            for pt in corners:
                cv2.circle(vis, tuple(pt), self.point_radius, self.neck_color, -1)

        # ── 2. Bund vonalak visszavetítve a kanonikus térből ───────────────
        H_inv = points.get("H_inv")
        fit = points.get("fit")
        if H_inv is not None and fit is not None:
            pred_x: dict = fit.get("predicted_x", {})
            fret_thickness = self._resolve_line_thickness(vis)
            for fret_n, fx in pred_x.items():
                fx = float(fx)
                # A bund egy függőleges vonal a kanonikus térben (x=fx, y=0..H)
                pt_top = np.array([fx, 0.0, 1.0])
                pt_bot = np.array([fx, float(CANONICAL_H), 1.0])
                proj_top = H_inv @ pt_top
                proj_bot = H_inv @ pt_bot
                if abs(proj_top[2]) < 1e-9 or abs(proj_bot[2]) < 1e-9:
                    continue
                tx = int(round(proj_top[0] / proj_top[2]))
                ty = int(round(proj_top[1] / proj_top[2]))
                bx = int(round(proj_bot[0] / proj_bot[2]))
                by = int(round(proj_bot[1] / proj_bot[2]))
                self._draw_outlined_line(
                    vis, (tx, ty), (bx, by),
                    color=self.fret_color,
                    thickness=fret_thickness,
                    outline_thickness=fret_thickness + 2,
                )
                # Minden 3. bund számozva
                if int(fret_n) % 3 == 0 and int(fret_n) > 0:
                    lx = (tx + bx) // 2
                    ly = min(ty, by) - 6
                    cv2.putText(
                        vis, str(int(fret_n)), (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                        self.fret_color, 1, cv2.LINE_AA,
                    )

        # ── 3. Nut oldal jelzése ───────────────────────────────────────────
        nut_side = direction or points.get("nut_side_hint")
        if nut_side is not None:
            nut_info = points.get("nut") or {}
            width_px = nut_info.get("width_px")
            nut_label = (f"nut: {nut_side} w={width_px:.1f}px"
                         if width_px is not None else f"nut: {nut_side}")
            cv2.putText(
                vis, nut_label, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                self.neck_color, 1, cv2.LINE_AA,
            )

        return vis

    # ──────────────────────────────────────────────────────────────────────
    # draw_landmarks
    # ──────────────────────────────────────────────────────────────────────

    def draw_landmarks(
        self,
        image: np.ndarray,
        hand_landmarks: list,
    ) -> np.ndarray:
        """MediaPipe 21 landmark pont és ujj-csontváz kapcsolatok berajzolása.

        A csontváz topológiát a ``src.constants.FINGER_CHAINS`` határozza meg
        (nem kerül újra implementálásra ebben a modulban).  Ujjhegy pontok
        (``FINGER_TIP_IDX``) nagyobb és eltérő színű körrel jelennek meg.

        Args:
            image:          BGR numpy ndarray.
            hand_landmarks: 21 elemű lista ``(x_norm, y_norm, z_norm)`` tuple-ökkel
                            (MediaPipe normalizált formátum, [0..1]).

        Returns:
            BGR numpy ndarray – landmark overlay-es kép.
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return image.copy()

        vis = image.copy()
        h, w = vis.shape[:2]

        # Pixel-koordináták kiszámítása
        pts_px: list[tuple[int, int]] = [
            (int(round(xn * w)), int(round(yn * h)))
            for xn, yn, _ in hand_landmarks
        ]

        # ── Csontváz vonalak (FINGER_CHAINS + kéztőcsontváz) ──────────────
        for a, b in self._HAND_CONNECTIONS:
            if a < len(pts_px) and b < len(pts_px):
                cv2.line(
                    vis, pts_px[a], pts_px[b],
                    color=self.connection_color,
                    thickness=self.line_thickness,
                    lineType=cv2.LINE_AA,
                )

        # ── Landmark pontok ───────────────────────────────────────────────
        for idx, pt in enumerate(pts_px):
            is_tip = idx in FINGER_TIP_IDX
            r = self.point_radius + 2 if is_tip else self.point_radius
            color = self.fingertip_color if is_tip else self.landmark_color
            cv2.circle(vis, pt, r, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, pt, r + 1, (10, 10, 10), 1, lineType=cv2.LINE_AA)

        return vis

    # ──────────────────────────────────────────────────────────────────────
    # get_intermediate_plots
    # ──────────────────────────────────────────────────────────────────────

    def get_intermediate_plots(
        self,
        image: np.ndarray,
        finger_mask: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Éldetektálás és Hough-transzformáció köztes vizualizációk.

        A tényleges edge-detektálást és vonalkeresést kizárólag a
        ``src.geometry.step1_canny`` és ``src.geometry.step2_hough`` végzi –
        a matematikai logika nem kerül újraimplementálásra ebben az osztályban.

        Args:
            image:        BGR bemeneti kép.
            finger_mask:  Opcionális bináris maszk (uint8, 0/255) az ujj-
                          területek elnyomásához (ugyanaz, amit a fő pipeline
                          is használ – ``result['finger_mask']``).

        Returns:
            Dict BGR numpy ndarray értékekkel:

            ``'canny'``
                Canny élkép BGR-be konvertálva (fehér élek fekete háttéren).
            ``'canny_masked'``
                Élek ujjmaszk alkalmazása után; csak ha ``finger_mask`` megadott.
            ``'hough'``
                Eredeti kép a detektált Hough-vonalakkal felülírva.
        """
        # Delayed import – kerüli a körkörös importokat, ha a geometry.py-t
        # csak itt szükséges behúzni.
        from src.geometry import step1_canny, step2_hough

        edges_full = step1_canny(image)

        # Ujjmaszk alkalmazás (opcionális – ugyanaz a logika, mint a pipeline-ban)
        edges_proc = edges_full.copy()
        if finger_mask is not None:
            edges_proc[finger_mask > 0] = 0

        lines = step2_hough(image, edges_proc)
        hough_thickness = self._resolve_line_thickness(image)

        # ── Canny vizualizáció (grayscale → BGR) ─────────────────────────
        canny_bgr = cv2.cvtColor(edges_full, cv2.COLOR_GRAY2BGR)

        # ── Hough vizualizáció – vonalak az eredeti képre ─────────────────
        hough_vis = image.copy()
        for x1, y1, x2, y2 in lines:
            self._draw_outlined_line(
                hough_vis, (x1, y1), (x2, y2),
                color=self.hough_line_color,
                thickness=hough_thickness,
                outline_thickness=hough_thickness + 2,
            )

        result: dict[str, np.ndarray] = {
            "canny": canny_bgr,
            "hough": hough_vis,
        }

        if finger_mask is not None:
            canny_masked_bgr = cv2.cvtColor(edges_proc, cv2.COLOR_GRAY2BGR)
            # Maszk-terület lila tintával jelölve
            tint = np.zeros_like(image)
            tint[finger_mask > 0] = [60, 10, 60]
            result["canny_masked"] = cv2.addWeighted(
                canny_masked_bgr, 1.0, tint, 0.45, 0
            )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Detektor-összehasonlítás
    # ──────────────────────────────────────────────────────────────────────

    def draw_detector_comparison(
        self,
        image: np.ndarray,
        result_geo: dict,
        result_int: dict,
        figsize: Optional[tuple] = None,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> "plt.Figure":
        """Geometriai vs. intenzitás-alapú detektor összehasonlítása.

        Sor 1 – Eredeti képre visszavetítve:
          [Geometriai overlay] | [Intenzitás overlay] | [Diff (csak eltérő bundok)]

        Sor 2 – Kanonikus tér:
          [Geo kanonikus + bundvonalak] | [Intenzitás kanonikus + bundvonalak]
          | [Gradiens-profil + csúcsok]

        Args:
            image:       Eredeti BGR kép.
            result_geo:  ``run_v14_pipeline(..., fret_detector=GeometricFretDetector())`` kimenet.
            result_int:  ``run_v14_pipeline(..., fret_detector=IntensityFretDetector())`` kimenet.
            figsize:     Matplotlib figsize.
            save_path:   Opcionális PNG mentés.

        Returns:
            Matplotlib Figure.
        """
        fig_size, scale = self._resolve_figsize_with_scale(
            figsize, self._figure_size_for_image(image, cols=3, rows=2)
        )
        fig, axes = plt.subplots(2, 3, figsize=fig_size, constrained_layout=True)
        fig.suptitle(
            "Detektor összehasonlítás: Geometriai vs. Intenzitás",
            fontsize=self._font_size(13, scale, minimum=10),
            fontweight="bold",
        )

        cls = result_geo.get("class", result_int.get("class", "?"))

        # ── Sor 1: visszavetített bundok az eredeti képen ─────────────────
        viz_geo = self.draw_fretboard_overlay(image, result_geo)
        viz_int = self.draw_fretboard_overlay(image, result_int)

        # Diff: bundok amelyek csak az egyik detektornál jelennek meg
        pred_geo = set(result_geo.get("fit", {}).get("predicted_x", {}).keys())
        pred_int = set(result_int.get("fit", {}).get("predicted_x", {}).keys())
        only_geo = pred_geo - pred_int
        only_int = pred_int - pred_geo
        common   = pred_geo & pred_int

        viz_diff = image.copy()
        px_geo = result_geo.get("fit", {}).get("predicted_x", {})
        px_int = result_int.get("fit", {}).get("predicted_x", {})
        H_inv_geo = result_geo.get("H_inv")
        H_inv_int = result_int.get("H_inv")
        for fret_n, fx in px_geo.items():
            color = (50, 200, 50) if fret_n in common else (0, 0, 220)
            if H_inv_geo is not None:
                self._draw_fret_line_on_image(viz_diff, H_inv_geo, float(fx), color)
        for fret_n, fx in px_int.items():
            if fret_n not in common:
                if H_inv_int is not None:
                    self._draw_fret_line_on_image(viz_diff, H_inv_int, float(fx),
                                                   (220, 100, 0))

        for ax, vis, title in [
            (axes[0, 0], viz_geo, f"Geometriai  [{cls}]  "
             f"cov={result_geo.get('fit', {}).get('coverage_ratio', 0):.2f}"),
            (axes[0, 1], viz_int, f"Intenzitás  [{cls}]  "
             f"cov={result_int.get('fit', {}).get('coverage_ratio', 0):.2f}"),
            (axes[0, 2], viz_diff, f"Eltérések  közös={len(common)} "
             f"csak-geo={len(only_geo)} csak-int={len(only_int)}"),
        ]:
            ax.imshow(vis[:, :, ::-1], interpolation="bilinear")
            self._set_equal_aspect(ax)
            ax.set_title(title, fontsize=self._font_size(9, scale, minimum=7))
            ax.axis("off")

        # ── Sor 2: kanonikus képek + gradiens-profil ──────────────────────
        for col, r, label in [
            (0, result_geo, "Geo – kanonikus"),
            (1, result_int, "Intenzitás – kanonikus"),
        ]:
            ax = axes[1, col]
            canon = r.get("canon")
            if canon is not None:
                vis2 = canon.copy()
                fit = r.get("fit")
                if fit:
                    fret_thickness = self._resolve_line_thickness(vis2)
                    for fret_n, fx in fit.get("predicted_x", {}).items():
                        xi = int(round(float(fx)))
                        self._draw_outlined_line(
                            vis2, (xi, 0), (xi, CANONICAL_H),
                            color=(80, 80, 240),
                            thickness=fret_thickness,
                            outline_thickness=fret_thickness + 2,
                        )
                        if int(fret_n) % 4 == 0 and int(fret_n) > 0:
                            cv2.putText(vis2, str(int(fret_n)),
                                        (xi + 2, CANONICAL_H - 3),
                                        cv2.FONT_HERSHEY_PLAIN, 0.55,
                                        (200, 200, 80), 1)
                ax.imshow(vis2[:, :, ::-1], interpolation="bilinear",
                          extent=[0, CANONICAL_W, CANONICAL_H, 0])
                self._set_equal_aspect(ax)
            else:
                ax.set_facecolor("#f8e8e8")
            ax.set_title(label, fontsize=self._font_size(9, scale, minimum=7))
            ax.axis("off")

        # Gradiens-profil (IntensityFretDetector-tól)
        ax_p = axes[1, 2]
        profile = result_int.get("intensity_profile")
        if profile is not None:
            xs = np.arange(len(profile))
            ax_p.fill_between(xs, profile, alpha=0.35, color="steelblue")
            ax_p.plot(xs, profile, color="steelblue", lw=self._line_width(self.line_thickness, scale, minimum=1.0))
            # Detektált csúcsok bejelölve
            for fx in result_int.get("fret_xs_raw", []):
                xi = int(round(fx))
                if 0 <= xi < len(profile):
                    ax_p.axvline(xi, color="C3", lw=self._line_width(self.line_thickness, scale, minimum=1.0), alpha=0.7)
            ax_p.set_xlim(0, len(profile))
            ax_p.set_ylim(0, 1.05)
            ax_p.set_xlabel("Kanonikus x (px)", fontsize=self._font_size(8, scale, minimum=7))
            ax_p.set_ylabel("Norm. gradiens", fontsize=self._font_size(8, scale, minimum=7))
        else:
            ax_p.text(0.5, 0.5, "Profil nem elérhető", ha="center", va="center",
                      transform=ax_p.transAxes, color="gray")
        ax_p.set_title(
            "Intenzitás gradiens-profil (piros = csúcs)",
            fontsize=self._font_size(9, scale, minimum=7),
        )
        ax_p.grid(True, alpha=0.3)
        ax_p.tick_params(labelsize=self._font_size(7, scale, minimum=6))

        self._enable_constrained_layout(fig)
        if save_path is not None:
            _save_figure(fig, save_path)
        if show:
            plt.show()
        return fig

    def _draw_fret_line_on_image(
        self,
        image: np.ndarray,
        H_inv: np.ndarray,
        fx: float,
        color: tuple,
    ) -> None:
        """Egy bund visszavetítése a kanonikus térből az eredeti képre (in-place)."""
        pt_top = np.array([fx, 0.0, 1.0])
        pt_bot = np.array([fx, float(CANONICAL_H), 1.0])
        proj_top = H_inv @ pt_top
        proj_bot = H_inv @ pt_bot
        if abs(proj_top[2]) < 1e-9 or abs(proj_bot[2]) < 1e-9:
            return
        tx = int(round(proj_top[0] / proj_top[2]))
        ty = int(round(proj_top[1] / proj_top[2]))
        bx = int(round(proj_bot[0] / proj_bot[2]))
        by = int(round(proj_bot[1] / proj_bot[2]))
        thickness = self._resolve_line_thickness(image)
        self._draw_outlined_line(
            image, (tx, ty), (bx, by),
            color=color,
            thickness=thickness,
            outline_thickness=thickness + 2,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Összetett kép segédletek
    # ──────────────────────────────────────────────────────────────────────

    def make_phase_strip(
        self,
        image: np.ndarray,
        finger_mask: Optional[np.ndarray] = None,
        strip_height: Optional[int] = None,
    ) -> np.ndarray:
        """Canny + Hough fázisok egymás alá fűzve (köztes sáv notebookhoz).

        Args:
            image:        BGR forrás kép.
            finger_mask:  Opcionális ujjmaszk.
            strip_height: Ha megadott, mindkét fáziságrát erre a magasságra
                          méretezi (hasznos grid-elrendezésnél).

        Returns:
            BGR numpy ndarray – canny + hough vertikálisan összefűzve.
        """
        phases = self.get_intermediate_plots(image, finger_mask)
        top = phases.get("canny_masked", phases["canny"])
        bot = phases["hough"]

        h_img, w_img = image.shape[:2]
        target_h = strip_height or (h_img // 2)
        target_w = w_img

        top_r = cv2.resize(top, (target_w, target_h), interpolation=cv2.INTER_AREA)
        bot_r = cv2.resize(bot, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Elválasztó vonal
        sep = np.full((3, target_w, 3), 180, dtype=np.uint8)
        return np.vstack([top_r, sep, bot_r])


    def draw_3panel_comparison(
        self,
        image: np.ndarray,
        result_geo: dict,
        result_int: dict,
        figsize: Optional[tuple] = None,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> "plt.Figure":
        """Háromoszlopos összehasonlító nézet egy képhez.

        Oszlopok:
          1. Eredeti kép + MediaPipe ujjak (ha vannak landmarks)
          2. GEOMETRIC_RULE bund-illesztés overlay
          3. INTENSITY_DATA bund-illesztés overlay

        A cím soronként mutatja: osztály | ok-státusz | coverage_ratio | raw csúcsok száma

        Args:
            image:       Eredeti BGR kép.
            result_geo:  ``run_v14_pipeline(..., fret_detector=GeometricFretDetector())`` kimenet.
            result_int:  ``run_v14_pipeline(..., fret_detector=IntensityFretDetector())`` kimenet.
            figsize:     Matplotlib figsize.
            save_path:   Opcionális PNG mentés.

        Returns:
            Matplotlib Figure.
        """
        fig_size, scale = self._resolve_figsize_with_scale(
            figsize, self._figure_size_for_image(image, cols=3, rows=1)
        )
        cls      = result_geo.get("class", result_int.get("class", "?"))
        fname    = result_geo.get("fname", result_geo.get("filename", ""))

        def _cov(r: dict) -> float:
            return r.get("fit", {}).get("coverage_ratio", 0.0) or 0.0

        def _n_raw(r: dict) -> int:
            xs = r.get("fret_xs_raw")
            return len(xs) if xs is not None else 0

        def _ok_str(r: dict) -> str:
            return "✓" if r.get("ok") else f"✗ {r.get('invalid_reason', '')}"

        # ── Panel képek összeállítása ──────────────────────────────────────
        vis_orig = self.draw_landmarks(image, result_geo.get("landmarks") or [])
        vis_geo  = self.draw_fretboard_overlay(image, result_geo)
        vis_int  = self.draw_fretboard_overlay(image, result_int)

        fig, axes = plt.subplots(1, 3, figsize=fig_size, constrained_layout=True)
        fig.suptitle(
            f"[{cls}]  {fname}",
            fontsize=self._font_size(10, scale, minimum=8),
            fontweight="bold",
        )

        panels = [
            (vis_orig,
             f"Eredeti + MediaPipe ujjak"),
            (vis_geo,
             f"GEOMETRIC_RULE  {_ok_str(result_geo)}\n"
             f"cov={_cov(result_geo):.3f}  raw_n={_n_raw(result_geo)}"),
            (vis_int,
             f"INTENSITY_DATA  {_ok_str(result_int)}\n"
             f"cov={_cov(result_int):.3f}  raw_n={_n_raw(result_int)}"),
        ]

        for ax, (vis, title) in zip(axes, panels):
            ax.imshow(vis[:, :, ::-1], interpolation="bilinear")
            self._set_equal_aspect(ax)
            ax.set_title(title, fontsize=self._font_size(8.5, scale, minimum=7))
            ax.axis("off")

        self._enable_constrained_layout(fig)
        if save_path is not None:
            _save_figure(fig, save_path)
        if show:
            plt.show()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Standalone segédfüggvények (backward-compatible API)
# ─────────────────────────────────────────────────────────────────────────────

def draw_pipeline_result(
    result: dict,
    figsize: Optional[tuple] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Egy pipeline result dict kétpaneles vizualizálása.

    Bal panel: eredeti kép + trapéz overlay.
    Jobb panel: kanonikus (600×80 px) kép + bund vonalak + ujjhegy pontok.
    """
    viz = PipelineVisualizer()
    img_bgr = result.get("img")
    fallback = viz._figure_size_for_image(img_bgr, cols=2, rows=1) if img_bgr is not None else viz._figure_size_for_canonical(cols=2, rows=1)
    fig_size, scale = viz._resolve_figsize_with_scale(figsize, fallback)
    fig, axes = plt.subplots(1, 2, figsize=fig_size, constrained_layout=True)
    cls = result.get("class", "?")
    ok = result.get("ok", False)
    fig.suptitle(
        f"[{cls}]  {result.get('fname', result.get('filename', ''))}  |  "
        f"{'OK' if ok else 'FAIL: ' + str(result.get('invalid_reason', ''))}",
        fontsize=viz._font_size(11, scale, minimum=8),
    )

    ax1 = axes[0]
    if img_bgr is not None:
        vis = viz.draw_fretboard_overlay(img_bgr, result)
        landmarks = result.get("landmarks")
        if landmarks:
            vis = viz.draw_landmarks(vis, landmarks)
        ax1.imshow(vis[:, :, ::-1], interpolation="bilinear")
    else:
        ax1.text(0.5, 0.5, "Kép nem elérhető", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=12, color="gray")
        ax1.set_facecolor("#f0f0f0")
    ax1.set_title("Eredeti kép + trapéz overlay")
    ax1.axis("off")

    ax2 = axes[1]
    canon = result.get("canon")
    if canon is not None and ok:
        fit = result.get("fit")
        vis2 = canon.copy()
        if fit is not None:
            for fret_n, fx in fit.get("predicted_x", {}).items():
                xi = int(round(float(fx)))
                viz._draw_outlined_line(
                    vis2, (xi, 0), (xi, CANONICAL_H),
                    color=(100, 100, 255),
                    thickness=viz._resolve_line_thickness(vis2),
                    outline_thickness=viz._resolve_line_thickness(vis2) + 2,
                )
                if int(fret_n) % 3 == 0:
                    cv2.putText(vis2, str(fret_n), (xi + 2, CANONICAL_H - 4),
                                cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 100), 1)
        for ft in result.get("fingertips", []):
            cx, cy = int(round(ft["canon_x"])), int(round(ft["canon_y"]))
            cv2.circle(vis2, (cx, cy), 5, (0, 255, 100), -1)
        ax2.imshow(vis2[:, :, ::-1], aspect="auto", interpolation="bilinear",
                   extent=[0, CANONICAL_W, CANONICAL_H, 0])
        coverage = (fit or {}).get("coverage_ratio", 0.0)
        ax2.set_title(f"Kanonikus tér  |  coverage={coverage:.2f}")
    else:
        ax2.set_facecolor("#f8e8e8" if not ok else "#f0f0f0")
        ax2.set_title("Kanonikus tér – nem elérhető")
    ax2.axis("off")

    viz._enable_constrained_layout(fig)
    if save_path is not None:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    return fig


def draw_pipeline_grid(
    results: list[dict],
    n_cols: int = 3,
    figsize_per: tuple = (6, 1.8),
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Több pipeline eredmény kanonikus képei grid-ben (batch diagnosztika)."""
    n = len(results)
    n_rows = (n + n_cols - 1) // n_cols
    vis = PipelineVisualizer()
    fallback = (max(6.0, figsize_per[0] * n_cols), max(4.0, figsize_per[1] * n_rows))
    fig_size, scale = vis._resolve_figsize_with_scale(None, fallback)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, constrained_layout=True)
    axes_flat = np.array(axes).flatten()
    for i, r in enumerate(results):
        ax = axes_flat[i]
        canon = r.get("canon") if r.get("ok") else None
        if canon is not None:
                ax.imshow(canon[:, :, ::-1], interpolation="bilinear")
                vis._set_equal_aspect(ax)
        else:
            ax.set_facecolor("#f8e8e8" if not r.get("ok") else "#f0f0f0")
        ax.set_title(f"[{r.get('class','?')}] {'OK' if r.get('ok') else 'FAIL'}\n"
                     f"{r.get('filename', r.get('fname', ''))}", fontsize=vis._font_size(7, scale, minimum=6))
        ax.axis("off")
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    vis._enable_constrained_layout(fig)
    if save_path is not None:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_training_history(
    history: list[dict],
    title: str = "",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 4),
    show: bool = True,
) -> plt.Figure:
    """Tanítási és validációs Loss/Accuracy görbék Phase-A / Phase-B jelöléssel."""
    phase_a = [h for h in history if h["phase"] == "A"]
    phase_b = [h for h in history if h["phase"] == "B"]
    viz = PipelineVisualizer()
    fig_size, scale = viz._resolve_figsize_with_scale(figsize, figsize)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=fig_size, constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=viz._font_size(13, scale, minimum=10), fontweight="bold")

    eps_a = [h["ep"] for h in phase_a]
    eps_b = [len(phase_a) + h["ep"] for h in phase_b]

    for ax, key, ylabel in [(ax_loss, "loss", "CrossEntropy Loss"),
                            (ax_acc,  "acc",  "Accuracy")]:
        if phase_a:
            ax.plot(eps_a, [h[f"tr_{key}"] for h in phase_a],
                "C0--", lw=viz._line_width(VIS_LINE_THICKNESS, scale, minimum=1.0), alpha=0.7, label="Train (A)")
            ax.plot(eps_a, [h[f"vl_{key}"] for h in phase_a],
                "C0-",  lw=viz._line_width(VIS_LINE_THICKNESS, scale, minimum=1.0), label="Val (A)")
        if phase_b:
            ax.plot(eps_b, [h[f"tr_{key}"] for h in phase_b],
                "C1--", lw=viz._line_width(VIS_LINE_THICKNESS, scale, minimum=1.0), alpha=0.7, label="Train (B)")
            ax.plot(eps_b, [h[f"vl_{key}"] for h in phase_b],
                "C1-",  lw=viz._line_width(VIS_LINE_THICKNESS, scale, minimum=1.0), label="Val (B)")
        if phase_a and phase_b:
            ax.axvline(len(phase_a) + 0.5, color="gray", ls=":", lw=viz._line_width(VIS_LINE_THICKNESS, scale, minimum=1.0), alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=viz._font_size(8, scale, minimum=7))
        ax.grid(True, alpha=0.3)

    if history:
        best_val = max(h["vl_acc"] for h in history)
        ax_acc.axhline(best_val, color="red", ls="--", lw=viz._line_width(VIS_LINE_THICKNESS, scale, minimum=1.0), alpha=0.5,
                       label=f"Best val={best_val:.3f}")
        ax_acc.set_ylim(0.0, 1.05)
        ax_acc.legend(fontsize=viz._font_size(8, scale, minimum=7))

    fig.set_constrained_layout(True)
    if save_path is not None:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_multi_training_histories(
    histories: dict[str, list[dict]],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
    show: bool = True,
) -> plt.Figure:
    """Több modell validációs accuracy / loss görbéjének összehasonlítása."""
    viz = PipelineVisualizer()
    fig_size, scale = viz._resolve_figsize_with_scale(figsize, figsize)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=fig_size, constrained_layout=True)
    fig.suptitle("Modellek összehasonlítása", fontsize=viz._font_size(13, scale, minimum=10), fontweight="bold")
    for i, (name, history) in enumerate(histories.items()):
        eps = list(range(1, len(history) + 1))
        ax_loss.plot(eps, [h["vl_loss"] for h in history], f"C{i}-", lw=viz._line_width(2, scale, minimum=1.0), label=name)
        ax_acc.plot(eps,  [h["vl_acc"]  for h in history], f"C{i}-", lw=viz._line_width(2, scale, minimum=1.0), label=name)
    for ax, ylabel in [(ax_loss, "Validation Loss"), (ax_acc, "Validation Accuracy")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=viz._font_size(9, scale, minimum=7))
        ax.grid(True, alpha=0.3)
    ax_acc.set_ylim(0.0, 1.05)
    fig.set_constrained_layout(True)
    if save_path is not None:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    return fig


def debug_nut_detection(
    result: dict,
    zoom_px: int = 100,
    figsize: tuple = (13, 8),
    show: bool = True,
) -> Optional[plt.Figure]:
    """3-panel Nut-diagnosztika egy run_v14_pipeline() result dictből.

    Panel 1 – teljes kanonikus kép (600×80 px) a detektált Nut oszlopával.
    Panel 2 – kinagyított Nut-környék (nyers képsáv) + függőleges intenzitás-profil.
    Panel 3 – Sobel-X col_response 1D jel: keresési ablak, küszöb, FWHM-jelölő.

    Opcionális debug tool – nem hívja az orchestrátort, csak result dictből olvas.

    Args:
        result:   run_v14_pipeline() kimenete.
        zoom_px:  ±hány px-t mutasson a Nut körül a 2. panelen.
        figsize:  matplotlib figsize.
        show:     plt.show() meghívása az ábrán (False = csak a Figure visszaadása).

    Returns:
        plt.Figure vagy None (ha nem volt kanonikus kép).
    """
    canon = result.get("canon")
    nut = result.get("nut")

    if canon is None:
        reason = result.get("invalid_reason", "ismeretlen")
        print(f"[debug_nut] Nincs kanonikus kép. invalid_reason: {reason}")
        return None

    W = canon.shape[1]  # CANONICAL_W = 600
    H_img = canon.shape[0]  # CANONICAL_H = 80

    # col_response: step6b_find_nut tárolja a result dictben
    col_response = nut.get("col_response") if nut else None
    if col_response is None:
        gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        col_response = np.abs(sx).sum(axis=0).astype(np.float32)

    nut_x = nut["nut_x"] if nut else None
    side_hint = result.get("nut_side_hint")
    hand_bnd_x = result.get("hand_boundary_canon_x")
    min_offset = 5

    # Keresési ablak határok (nominális – clamp nélkül, közelítő)
    if side_hint is not None:
        sw = max(int(W * 0.40), 10)
        if side_hint == "left":
            search_lo, search_hi = min_offset, min(sw, W - 1)
        else:
            search_lo, search_hi = max(0, W - sw), W - min_offset
    else:
        sw = max(int(W * 0.30), 10)
        search_lo_l, search_hi_l = min_offset, min(sw, W - 1)
        search_lo_r, search_hi_r = max(0, W - sw), W - min_offset
        search_lo, search_hi = search_lo_l, search_hi_r

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.2, 2.5, 3.0],
        width_ratios=[3.5, 1],
        hspace=0.55, wspace=0.28,
    )
    ax_full  = fig.add_subplot(gs[0, :])
    ax_zoom  = fig.add_subplot(gs[1, 0])
    ax_vprof = fig.add_subplot(gs[1, 1])
    ax_prof  = fig.add_subplot(gs[2, :])

    # ── Panel 1: teljes kanonikus kép ─────────────────────────────────────────
    ax_full.imshow(canon[:, :, ::-1], aspect="auto",
                   extent=[0, W, H_img, 0])
    if nut_x is not None:
        ax_full.axvline(nut_x, color="lime", lw=2.0, alpha=0.95,
                        label=f"Nut x={nut_x}px")
        ax_full.legend(fontsize=8, loc="lower right")
    if hand_bnd_x is not None:
        ax_full.axvline(hand_bnd_x, color="red", lw=1.2, ls="--", alpha=0.7,
                        label=f"Kézél={hand_bnd_x:.0f}px")
    ax_full.set_title(
        f"Kanonikus kép  |  osztály: {result.get('class', '?')}  |  "
        f"{'✓ Nut @ x=' + str(nut_x) + 'px' if nut_x is not None else '✗ Nut NEM detektált'}",
        fontsize=9,
    )
    ax_full.set_yticks([])
    ax_full.set_xlabel("Kanonikus x [px]", fontsize=8)
    ax_full.tick_params(labelsize=7)

    # ── Panel 2a: kinagyított Nut-sáv ─────────────────────────────────────────
    if nut_x is not None:
        lo = max(0, nut_x - zoom_px)
        hi = min(W, nut_x + zoom_px)
        strip = canon[:, lo:hi]
        ax_zoom.imshow(strip[:, :, ::-1], aspect="auto",
                       extent=[lo, hi, H_img, 0])
        ax_zoom.axvline(nut_x, color="lime", lw=1.8, alpha=0.9)
        ax_zoom.set_title(f"Nut-környék (±{zoom_px}px)", fontsize=9)
        ax_zoom.set_xlabel("Kanonikus x [px]", fontsize=8)
        ax_zoom.tick_params(labelsize=7)
    else:
        ax_zoom.text(0.5, 0.5, "Nincs Nut-találat\na keresési tartományban",
                     ha="center", va="center", transform=ax_zoom.transAxes,
                     color="red", fontsize=10)
        ax_zoom.set_facecolor("#fff0f0")
        ax_zoom.axis("off")

    # ── Panel 2b: függőleges intenzitás-profil a Nut oszlopán ─────────────────
    if nut_x is not None:
        gray_c = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
        x_lo = max(0, nut_x - 2)
        x_hi = min(W, nut_x + 3)
        col_strip = gray_c[:, x_lo:x_hi].mean(axis=1)
        ys = np.arange(len(col_strip))
        ax_vprof.plot(col_strip, ys, color="steelblue", lw=1.6)
        ax_vprof.invert_yaxis()
        ax_vprof.set_title(f"Intenzitás\n@ x={nut_x}", fontsize=8)
        ax_vprof.set_xlabel("Intenz.", fontsize=7)
        ax_vprof.set_ylabel("y [px]", fontsize=7)
        ax_vprof.tick_params(labelsize=6)
        ax_vprof.grid(alpha=0.3)
        ax_vprof.set_ylim(H_img - 0.5, -0.5)
    else:
        ax_vprof.axis("off")

    # ── Panel 3: 1D col_response profil ──────────────────────────────────────
    xs = np.arange(len(col_response))
    ax_prof.plot(xs, col_response, color="steelblue", lw=1.3, alpha=0.85,
                 label="Sobel-X |oszlopválasz|")
    ax_prof.fill_between(xs, col_response, alpha=0.15, color="steelblue")

    # Keresési ablak(ok) árnyékolva
    if side_hint == "left":
        ax_prof.axvspan(search_lo, search_hi, alpha=0.13, color="green",
                        label=f"Keresési ablak (bal  {search_lo}–{search_hi}px, ~40%)")
    elif side_hint == "right":
        ax_prof.axvspan(search_lo, search_hi, alpha=0.13, color="green",
                        label=f"Keresési ablak (jobb {search_lo}–{search_hi}px, ~40%)")
    else:
        ax_prof.axvspan(search_lo_l, search_hi_l, alpha=0.13, color="green",
                        label=f"Keresési ablak (bal+jobb, ~30%)")
        ax_prof.axvspan(search_lo_r, search_hi_r, alpha=0.13, color="green")

    # Kézél határvonal
    if hand_bnd_x is not None:
        ax_prof.axvline(hand_bnd_x, color="red", lw=1.2, ls="--", alpha=0.7,
                        label=f"Kézél={hand_bnd_x:.0f}px (keresés korlátozás)")

    # Küszöb vonalak
    median_r = float(np.median(col_response))
    ax_prof.axhline(median_r * 2.5, color="darkorange", lw=1.1, ls="--", alpha=0.8,
                    label=f"Küszöb 2.5× medián ({median_r * 2.5:.0f})")
    ax_prof.axhline(median_r * 2.0, color="gold", lw=0.9, ls=":", alpha=0.6,
                    label=f"Küszöb 2.0× medián ({median_r * 2.0:.0f}) [side_hint]")

    # Detektált Nut jelölő + FWHM nyíl
    if nut_x is not None:
        peak_val = float(col_response[nut_x])
        fwhm = float(nut.get("width_px", 0))
        ratio = float(nut.get("ratio", 0))
        ax_prof.axvline(nut_x, color="lime", lw=2.2,
                        label=f"Nut x={nut_x}px  csúcs={peak_val:.0f}  arány={ratio:.2f}  FWHM={fwhm:.1f}px")
        # FWHM jelölő nyíl
        if fwhm > 0:
            half_val = peak_val * 0.5
            ax_prof.annotate(
                "", xy=(nut_x + fwhm / 2, half_val),
                xytext=(nut_x - fwhm / 2, half_val),
                arrowprops=dict(arrowstyle="<->", color="lime", lw=1.5),
            )
            ax_prof.text(nut_x, half_val * 1.05, f"FWHM={fwhm:.1f}px",
                         ha="center", va="bottom", fontsize=7, color="lime")

    ax_prof.set_xlabel("Kanonikus x [px]", fontsize=9)
    ax_prof.set_ylabel("Sobel-X oszlopválasz (összeg)", fontsize=9)
    ax_prof.set_title("step6b_find_nut — 1D Sobel-X profil", fontsize=9)
    ax_prof.legend(fontsize=7.5, loc="upper right")
    ax_prof.grid(alpha=0.25)
    ax_prof.set_xlim(0, W)

    # Cím
    nut_summary = (
        f"side={nut['side']}  ratio={nut['ratio']:.2f}  FWHM={nut.get('width_px', 0):.1f}px"
        if nut else "NEM DETEKTÁLT"
    )
    fig.suptitle(
        f"Nut-diagnosztika  —  {result.get('fname', '?')}\n"
        f"side_hint={side_hint}  |  {nut_summary}",
        fontsize=10, y=1.01,
    )

    if show:
        plt.show()
    return fig


def plot_scatter_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    title: str = "",
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
    show: bool = True,
) -> plt.Figure:
    """2D scatter plot osztályonkénti színezéssel (PCA / t-SNE kimenethez)."""
    viz = PipelineVisualizer()
    fig_size, scale = viz._resolve_figsize_with_scale(figsize, figsize)
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    for idx, cls_name in enumerate(classes):
        mask = labels == idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=class_color_mpl(idx), label=cls_name,
                   s=60, alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.set_title(title, fontsize=viz._font_size(12, scale, minimum=9))
    ax.legend(title="Akkord", fontsize=viz._font_size(9, scale, minimum=7), bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.set_constrained_layout(True)
    if save_path is not None:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    return fig
