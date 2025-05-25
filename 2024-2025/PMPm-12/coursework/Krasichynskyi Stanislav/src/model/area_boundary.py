import numpy as np

## TODO. Improve class
class AreaBoundary:
    def __init__(self) -> None:
        self.points = np.empty((0, 2), dtype=np.float32)
        self.segments = np.empty((0, 2), dtype=np.int32)
