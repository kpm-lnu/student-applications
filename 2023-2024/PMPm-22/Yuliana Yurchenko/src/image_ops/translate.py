from src.image_ops.base import BaseImageOperation
from src.image_ops.utils.transformations import translate


class TranslateOperation(BaseImageOperation):
    """
    Class that implements operation of translating an image
    """
    def __init__(self, twidth, theight):
        self._op = lambda X: translate(X, twidth, theight)