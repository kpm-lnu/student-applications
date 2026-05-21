from __future__ import annotations

from dataclasses import dataclass
from math import gamma
from typing import Callable

import numpy as np

ScalarOrArray = np.ndarray | float


@dataclass(frozen=True, slots=True)
class FractionalProblemConfig:
    s: float = 0.75
    gamma_mesh: float = 4.0
    L: float = 1.0
    Y: float = 5.0
    mesh_sizes: tuple[int, ...] = (16, 32, 64, 128)

    @property
    def alpha(self) -> float:
        return 1.0 - 2.0 * self.s

    @property
    def d_s(self) -> float:
        return (2.0 ** (1.0 - 2.0 * self.s)) * (gamma(1.0 - self.s) / gamma(self.s))


@dataclass(slots=True)
class FractionalProblem:
    config: FractionalProblemConfig
    source_function: Callable[[ScalarOrArray], ScalarOrArray] | None = None

    def source(self, x: ScalarOrArray) -> ScalarOrArray:
        if self.source_function is not None:
            return self.source_function(x)
        return np.sin(np.pi * x)

    def exact_solution(self, x: ScalarOrArray) -> ScalarOrArray:
        return (1.0 / ((np.pi ** 2) ** self.config.s)) * np.sin(np.pi * x)

    def exact_solution_derivative(self, x: ScalarOrArray) -> ScalarOrArray:
        return (np.pi / ((np.pi ** 2) ** self.config.s)) * np.cos(np.pi * x)
