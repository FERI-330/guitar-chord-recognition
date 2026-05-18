#!/usr/bin/env python3
"""
Refactor 04_5 through 04_10 notebooks:
1. Replace the big setup cell with imports from src modules
2. Delete cells with function definitions
3. Keep visualization functions
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOKS_TO_REFACTOR = [
    "04_5_neck_angle.ipynb",
    "04_6_line_split.ipynb",
    "04_7_outer_edges.ipynb",
    "04_8_trapezoid.ipynb",
    "04_9_canonical_nut_frets.ipynb",
    "04_10_spacing_fit.ipynb",
]

IMPORTS_CELL = """from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
    print(f"cv2 elérhető: {cv2.__version__}")
except Exception:
    cv2 = None
    CV2_AVAILABLE = False
    print("⚠️  cv2 NEM elérhető – a legtöbb lépés kihagyódik.")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
    print("mediapipe elérhető")
except Exception:
    mp = mp_python = mp_vision = None
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  mediapipe NEM elérhető – kézdetektálás kihagyható.")

# ── Import dari src modules ──────────────────────────────────────────────────
from src.hand_landmark import build_finger_mask, anchor_neck_angle, FINGER_CHAINS, FINGER_THICK_MULT
from src.geometry import step1_canny, step2_hough, step3_neck_angle, step3_neck_angle_anchored, step4_split_lines, step5_outer_edges

# ── Projekt útvonalak ─────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "data").exists() and (PROJECT_ROOT.parent / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_ROOT     = PROJECT_ROOT / "data"
NOTEBOOK_DIR  = PROJECT_ROOT / "notebooks"
MODEL_DIR     = PROJECT_ROOT / "models"
MANIFEST_PATH = DATA_ROOT / "split_manifest.csv"
OUTPUT_DIR    = PROJECT_ROOT / "output" / "03b_pipeline_debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Geometriai konstansok ──────────────────────────────────────────────────
CANONICAL_W   = 600          # kanonikus tér szélessége (px)
CANONICAL_H   = 80           # kanonikus tér magassága (px)
N_FRETS       = 24
FRET_RULE     = 17.817

# Bund n x-pozíciója ha a teljes nyak kitölti a kanonikus teret [0..N_FRETS]
# x_n = CANONICAL_W * (1 - 0.5^(n/12)) / 0.75
FRET_POS_FULL = np.array(
    [CANONICAL_W * (1.0 - 0.5 ** (n / 12.0)) / 0.75 for n in range(N_FRETS + 1)],
    dtype=np.float64,
)
# Normalizált pozíciók (0..1, ahol 1.0 = a 24. bund pozíciója)
FRET_POS_NORM = np.array(
    [1.0 - 0.5 ** (n / 12.0) for n in range(N_FRETS + 1)],
    dtype=np.float64,
)

FINGER_TIP_IDX = [4, 8, 12, 16, 20]   # MediaPipe ujjhegy landmark indexek
INLAY_FRETS    = [3, 5, 7, 9, 12]     # Standard 5 inlay pozíció
INLAY_FRETS_FULL = [3, 5, 7, 9, 12, 15, 17, 19, 21, 24]  # Mindkét oktáv

# Inlay normalizált x-pozíciók: az n. inlay a FRET_POS_NORM[n-1] és [n] közötti tér közepe
INLAY_NORM_DICT = {
    n: (FRET_POS_NORM[n - 1] + FRET_POS_NORM[n]) / 2.0
    for n in INLAY_FRETS_FULL
}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print(f"Project root : {PROJECT_ROOT}")
print(f"Output dir   : {OUTPUT_DIR}")
print(f"FRET_POS_FULL[:7] = {np.round(FRET_POS_FULL[:7], 1)}")"""

FUNCTION_DEFINITIONS_TO_DELETE = [
    "FINGER_CHAINS",
    "FINGER_THICK_MULT",
    "def build_finger_mask",
    "def anchor_neck_angle",
    "def _normalize_angle",
    "def step1_canny",
    "def step2_hough",
    "def step3_neck_angle",
    "def step3_neck_angle_anchored",
    "def step4_split_lines",
    "def step5_outer_edges",
    "def _fit_cluster_edge",
    "def _fit_extreme_edge",
    "def _find_neck_edge_outliers",
    "def _dbscan_1d",
    "def _line_stats",
]

def cell_contains_function_def(cell):
    """Check if cell contains function definitions that should be removed."""
    if cell.get("cell_type") != "code":
        return False
    source = "".join(cell.get("source", []))
    for func in FUNCTION_DEFINITIONS_TO_DELETE:
        if func in source:
            return True
    return False

def cell_is_viz_function(cell):
    """Check if cell is a visualization function that should be kept."""
    if cell.get("cell_type") != "code":
        return False
    source = "".join(cell.get("source", []))
    return "def viz_" in source or "def _draw_" in source

def should_keep_cell(cell, is_setup_cell=False):
    """Determine if a cell should be kept."""
    if is_setup_cell:
        return True
    if cell_is_viz_function(cell):
        return True
    if cell_contains_function_def(cell) and not cell_is_viz_function(cell):
        return False
    return True

def refactor_notebook(notebook_path):
    """Refactor a single notebook."""
    print(f"\nRefactoring: {notebook_path.name}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    new_cells = []
    setup_updated = False
    
    for i, cell in enumerate(nb.get("cells", [])):
        # First code cell (after markdowns) → replace with imports
        if not setup_updated and cell.get("cell_type") == "code":
            print(f"  - Updating setup cell (index {i})")
            cell["source"] = IMPORTS_CELL.split('\n')
            cell["source"] = [line + '\n' for line in cell["source"][:-1]] + [cell["source"][-1]]
            new_cells.append(cell)
            setup_updated = True
        # Skip cells with function definitions (except viz functions)
        elif cell_contains_function_def(cell) and not cell_is_viz_function(cell):
            print(f"  - Deleting function definition cell (index {i})")
            continue
        # Keep all other cells
        else:
            new_cells.append(cell)
    
    nb["cells"] = new_cells
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"  ✅ Refactored {notebook_path.name}")

if __name__ == "__main__":
    for nb_name in NOTEBOOKS_TO_REFACTOR:
        nb_path = NOTEBOOKS_DIR / nb_name
        if nb_path.exists():
            refactor_notebook(nb_path)
        else:
            print(f"⚠️  Not found: {nb_path}")
    print("\n✅ All notebooks refactored!")
