from dataclasses import dataclass
from typing import List
from datetime import datetime

from .insertion import Insertion


@dataclass
class Cycle:
    x: float
    y: float
    radius_x: float
    radius_y: float
    insertions: List[Insertion]

    def latest_insertion(self) -> Insertion:
        return max(self.insertions, key=lambda insertion: insertion.timestamp)

    def oldest_insertion(self) -> Insertion:
        return min(self.insertions, key=lambda insertion: insertion.timestamp)

    def latest_timestamp(self) -> int:
        return self.latest_insertion().timestamp

    def oldest_timestamp(self) -> Insertion:
        return self.oldest_insertion().timestamp


    def append(self, x: float, y: float) -> Insertion:
        offset_x = x - self.x
        offset_y = y - self.y
        insertion = Insertion(offset_x, offset_y, int(datetime.utcnow().timestamp()))
        self.insertions.append(insertion)
        return insertion

    @staticmethod
    def from_dict(data: dict):
        return Cycle(
            data['x'],
            data['y'],
            data['radius_x'],
            data['radius_y'],
            [Insertion.from_dict(insertion) for insertion in data['insertions']]
        )

    def to_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'radius_x': self.radius_x,
            'radius_y': self.radius_y,
            'insertions': [insertion.to_dict() for insertion in self.insertions]
        }
