from __future__ import annotations
import numpy as np
from src.config import CFG

# Direkten exportált konstansok (kényelem + visszafelé-kompatibilitás)
CANONICAL_W: int = CFG["canonical_w"]
CANONICAL_H: int = CFG["canonical_h"]
N_FRETS: int = CFG["n_frets"]

_W = CANONICAL_W
_N = N_FRETS

# x_n = CANONICAL_W * (1 - 0.5^(n/12)) / 0.75  [0..N_FRETS beleértve]
FRET_POS_FULL: np.ndarray = np.array(
    [_W * (1.0 - 0.5 ** (n / 12.0)) / 0.75 for n in range(_N + 1)],
    dtype=np.float64,
)

# Normalizált pozíciók [0..1]
FRET_POS_NORM: np.ndarray = np.array(
    [1.0 - 0.5 ** (n / 12.0) for n in range(_N + 1)],
    dtype=np.float64,
)

INLAY_FRETS: list[int] = [3, 5, 7, 9, 12]
INLAY_FRETS_FULL: list[int] = [3, 5, 7, 9, 12, 15, 17, 19, 21, 24]

# Inlay normalizált x-pozíciók: az inlay az n-1. és n. bund közötti középpont
INLAY_NORM_DICT: dict[int, float] = {
    n: float((FRET_POS_NORM[n - 1] + FRET_POS_NORM[n]) / 2.0)
    for n in INLAY_FRETS_FULL
}

FINGER_TIP_IDX: list[int] = [4, 8, 12, 16, 20]

# MediaPipe ujj-lánc topológia (csukló→MCP szándékosan kihagyva)
FINGER_CHAINS: dict[str, list[int]] = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# Ujjvastagság-szorzók (kéz-skálához relatív)
FINGER_THICK_MULT: dict[str, float] = {
    "thumb": 0.55, "index": 0.40, "middle": 0.40,
    "ring":  0.36, "pinky": 0.32,
}
