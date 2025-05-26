from dataclasses import dataclass


@dataclass
class Insertion:
    offset_x: float
    offset_y: float
    timestamp: int

    @staticmethod
    def from_dict(data: dict):
        return Insertion(
            data['offset_x'],
            data['offset_y'],
            data['timestamp']
        )

    def to_dict(self) -> dict:
        return {
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'timestamp': self.timestamp
        }
