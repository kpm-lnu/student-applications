from src.image_ops.base import BaseImageOperation
from src.image_ops.utils.transformations import scale


class ShiftOperation(BaseImageOperation):
    """
    Class that implements operation of scaling an image
    """
    def __init__(self, shift: float):
        self._op = lambda X: X + shift
