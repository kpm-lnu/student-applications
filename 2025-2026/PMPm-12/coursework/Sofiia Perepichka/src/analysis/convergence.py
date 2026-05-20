from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConvergenceRow:
    N: int
    h: float
    l2_error: float
    w21_error: float
    l2_order: float | None = None
    w21_order: float | None = None


class ConvergenceAnalyzer:
    @staticmethod
    def compute_order(previous_error: float, current_error: float) -> float:
        return float(np.log(previous_error / current_error) / np.log(2.0))

    def add_orders(self, rows: list[ConvergenceRow]) -> list[ConvergenceRow]:
        for i in range(1, len(rows)):
            rows[i].l2_order = self.compute_order(rows[i - 1].l2_error, rows[i].l2_error)
            rows[i].w21_order = self.compute_order(rows[i - 1].w21_error, rows[i].w21_error)

        return rows
