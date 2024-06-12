from dataclasses import dataclass
from typing import List
from math import sin, cos, pi
from random import random

from .cycle import Cycle


@dataclass
class Area:
    label: str
    cycles: List[Cycle]

    def get_latest_cycle(self) -> Cycle:
        return max(
            self.cycles,
            key=lambda cycle: cycle.latest_timestamp()
        )


    def add_cycle(self, cx: float, cy: float, rx: float, ry: float) -> Cycle:
        cycle = Cycle(cx, cy, rx, ry, [])
        self.cycles.append(cycle)
        return cycle

    def add_shifted_cycle(self,
                          prev_cycle: Cycle,
                          distance: float = 0.125) -> Cycle:
        theta = 2*pi*random()
        shift_x = distance * cos(theta)
        shift_y = distance * sin(theta)
        cx = prev_cycle.x + shift_x
        cy = prev_cycle.y + shift_y
        return self.add_cycle(cx, cy, prev_cycle.radius_x, prev_cycle.radius_y)


    @staticmethod
    def from_dict(data: dict):
        return Area(
            data['label'],
            [Cycle.from_dict(cycle) for cycle in data['cycles']]
        )

    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'cycles': [cycle.to_dict() for cycle in self.cycles]
        }
