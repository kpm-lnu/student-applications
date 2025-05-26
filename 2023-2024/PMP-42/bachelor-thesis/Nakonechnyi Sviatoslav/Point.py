from itertools import count
class Point:
    _id = count(0)

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.id = next(self._id)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __str__(self):
        return self.__repr__()
