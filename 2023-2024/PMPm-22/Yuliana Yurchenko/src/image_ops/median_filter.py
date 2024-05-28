from src.image_ops.base import BaseImageOperation
from src.image_ops.utils.blur_images import median_filter


class MedianFilterOperation(BaseImageOperation):
    """
    Class that implements operation of adding median filter to an image.
    """
    def __init__(self, radius=3):
        self._op = lambda X: median_filter(X, radius)

