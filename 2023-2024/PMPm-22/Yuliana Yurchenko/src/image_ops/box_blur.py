from src.image_ops.base import BaseImageOperation
from src.image_ops.utils.blur_images import box_blur


class BoxBlurOperation(BaseImageOperation):
    """
    Class that implements operation of adding box blur to an image.
    """
    def __init__(self, radius=1):
        self._op = lambda X: box_blur(X, radius)

