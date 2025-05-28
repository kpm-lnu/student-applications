from dataclasses import dataclass
from typing import List
import json

from .area import Area


@dataclass
class History:
    areas: List[Area]

    def get_area(self, label: str) -> Area:
        return next(area for area in self.areas if area.label == label)

    def __getitem__(self, area_label: str) -> Area:
        return self.get_area(area_label)

    def add_area(self, label: str) -> Area:
        area = Area(label, [])
        self.areas.append(area)
        return area

    @staticmethod
    def load_from_json_file(file_path: str):
        with open(file_path) as f:
            data = json.load(f)
        areas = [Area.from_dict(area) for area in data['areas']]
        return History(areas)

    def save_to_json_file(self, file_path: str) -> None:
        data = [area.to_dict() for area in self.areas]
        with open(file_path, 'w') as f:
            json.dump(data, f)
