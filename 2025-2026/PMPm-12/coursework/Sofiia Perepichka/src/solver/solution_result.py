from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SolutionResult:
    mesh: dict
    vertices: np.ndarray
    triangles: np.ndarray
    values: np.ndarray
