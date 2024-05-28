"""Request Body Class"""


from typing import Generic, TypeVar

from core.classes.model.model import Model


# pylint: disable=invalid-name
ParametersType = TypeVar('ParametersType')
# pylint: enable=invalid-name


class RequestBody(Generic[ParametersType]):
    """Request Body"""

    @property
    def definition(self) -> None:
        """Definition"""

        raise NotImplementedError

    parameters: ParametersType
    model: Model

    def __init__(self, parameters: ParametersType, model: Model) -> None:
        self.parameters = parameters
        self.model = model
