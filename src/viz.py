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

from src.config import CFG, PATHS
from src.constants import (
    CANONICAL_W, CANONICAL_H,
    FINGER_TIP_IDX, FINGER_CHAINS,
)


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

    def __init__(
        self,
        neck_color: tuple = (50, 220, 50),
        fret_color: tuple = (80, 80, 240),
        landmark_color: tuple = (0, 230, 230),
        connection_color: tuple = (230, 160, 0),
        fingertip_color: tuple = (0, 180, 255),
        line_thickness: int = 2,
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
                cv2.line(
                    vis, (tx, ty), (bx, by),
                    color=self.fret_color,
                    thickness=max(1, self.line_thickness - 1),
                    lineType=cv2.LINE_AA,
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
            cv2.putText(
                vis, f"nut: {nut_side}", (10, 24),
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

        # ── Canny vizualizáció (grayscale → BGR) ─────────────────────────
        canny_bgr = cv2.cvtColor(edges_full, cv2.COLOR_GRAY2BGR)

        # ── Hough vizualizáció – vonalak az eredeti képre ─────────────────
        hough_vis = image.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(
                hough_vis, (x1, y1), (x2, y2),
                color=self.hough_line_color,
                thickness=max(1, self.line_thickness - 1),
                lineType=cv2.LINE_AA,
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


# ─────────────────────────────────────────────────────────────────────────────
# Standalone segédfüggvények (backward-compatible API)
# ─────────────────────────────────────────────────────────────────────────────

def draw_pipeline_result(
    result: dict,
    figsize: tuple = (18, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Egy pipeline result dict kétpaneles vizualizálása.

    Bal panel: eredeti kép + trapéz overlay.
    Jobb panel: kanonikus (600×80 px) kép + bund vonalak + ujjhegy pontok.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    cls = result.get("class", "?")
    ok = result.get("ok", False)
    fig.suptitle(
        f"[{cls}]  {result.get('fname', result.get('filename', ''))}  |  "
        f"{'OK' if ok else 'FAIL: ' + str(result.get('invalid_reason', ''))}",
        fontsize=11,
    )

    viz = PipelineVisualizer()

    ax1 = axes[0]
    img_bgr = result.get("img")
    if img_bgr is not None:
        vis = viz.draw_fretboard_overlay(img_bgr, result)
        landmarks = result.get("landmarks")
        if landmarks:
            vis = viz.draw_landmarks(vis, landmarks)
        ax1.imshow(vis[:, :, ::-1])
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
                cv2.line(vis2, (xi, 0), (xi, CANONICAL_H), (100, 100, 255), 1)
                if int(fret_n) % 3 == 0:
                    cv2.putText(vis2, str(fret_n), (xi + 2, CANONICAL_H - 4),
                                cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 100), 1)
        for ft in result.get("fingertips", []):
            cx, cy = int(round(ft["canon_x"])), int(round(ft["canon_y"]))
            cv2.circle(vis2, (cx, cy), 5, (0, 255, 100), -1)
        ax2.imshow(vis2[:, :, ::-1], aspect="auto",
                   extent=[0, CANONICAL_W, CANONICAL_H, 0])
        coverage = (fit or {}).get("coverage_ratio", 0.0)
        ax2.set_title(f"Kanonikus tér  |  coverage={coverage:.2f}")
    else:
        ax2.set_facecolor("#f8e8e8" if not ok else "#f0f0f0")
        ax2.set_title("Kanonikus tér – nem elérhető")
    ax2.axis("off")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def draw_pipeline_grid(
    results: list[dict],
    n_cols: int = 3,
    figsize_per: tuple = (6, 1.8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Több pipeline eredmény kanonikus képei grid-ben (batch diagnosztika)."""
    n = len(results)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows))
    axes_flat = np.array(axes).flatten()
    for i, r in enumerate(results):
        ax = axes_flat[i]
        canon = r.get("canon") if r.get("ok") else None
        if canon is not None:
            ax.imshow(canon[:, :, ::-1], aspect="auto")
        else:
            ax.set_facecolor("#f8e8e8" if not r.get("ok") else "#f0f0f0")
        ax.set_title(f"[{r.get('class','?')}] {'OK' if r.get('ok') else 'FAIL'}\n"
                     f"{r.get('filename', r.get('fname', ''))}", fontsize=7)
        ax.axis("off")
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def plot_training_history(
    history: list[dict],
    title: str = "",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Tanítási és validációs Loss/Accuracy görbék Phase-A / Phase-B jelöléssel."""
    phase_a = [h for h in history if h["phase"] == "A"]
    phase_b = [h for h in history if h["phase"] == "B"]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    eps_a = [h["ep"] for h in phase_a]
    eps_b = [len(phase_a) + h["ep"] for h in phase_b]

    for ax, key, ylabel in [(ax_loss, "loss", "CrossEntropy Loss"),
                            (ax_acc,  "acc",  "Accuracy")]:
        if phase_a:
            ax.plot(eps_a, [h[f"tr_{key}"] for h in phase_a],
                    "C0--", lw=1.5, alpha=0.7, label="Train (A)")
            ax.plot(eps_a, [h[f"vl_{key}"] for h in phase_a],
                    "C0-",  lw=2.0, label="Val (A)")
        if phase_b:
            ax.plot(eps_b, [h[f"tr_{key}"] for h in phase_b],
                    "C1--", lw=1.5, alpha=0.7, label="Train (B)")
            ax.plot(eps_b, [h[f"vl_{key}"] for h in phase_b],
                    "C1-",  lw=2.0, label="Val (B)")
        if phase_a and phase_b:
            ax.axvline(len(phase_a) + 0.5, color="gray", ls=":", lw=1.2, alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    if history:
        best_val = max(h["vl_acc"] for h in history)
        ax_acc.axhline(best_val, color="red", ls="--", lw=1, alpha=0.5,
                       label=f"Best val={best_val:.3f}")
        ax_acc.set_ylim(0.0, 1.05)
        ax_acc.legend(fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_multi_training_histories(
    histories: dict[str, list[dict]],
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Több modell validációs accuracy / loss görbéjének összehasonlítása."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Modellek összehasonlítása", fontsize=13, fontweight="bold")
    for i, (name, history) in enumerate(histories.items()):
        eps = list(range(1, len(history) + 1))
        ax_loss.plot(eps, [h["vl_loss"] for h in history], f"C{i}-", lw=2, label=name)
        ax_acc.plot(eps,  [h["vl_acc"]  for h in history], f"C{i}-", lw=2, label=name)
    for ax, ylabel in [(ax_loss, "Validation Loss"), (ax_acc, "Validation Accuracy")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    ax_acc.set_ylim(0.0, 1.05)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_scatter_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    title: str = "",
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """2D scatter plot osztályonkénti színezéssel (PCA / t-SNE kimenethez)."""
    fig, ax = plt.subplots(figsize=figsize)
    for idx, cls_name in enumerate(classes):
        mask = labels == idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=class_color_mpl(idx), label=cls_name,
                   s=60, alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.set_title(title, fontsize=12)
    ax.legend(title="Akkord", fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig
