"""Success Response Class"""

from typing import Generic, TypeVar


# pylint: disable=invalid-name
ParametersType = TypeVar('ParametersType')
ResultType = TypeVar('ResultType')
# pylint: enable=invalid-name


class SuccessResponse(Generic[ParametersType, ResultType]):
    """Success response"""
    parameters: ParametersType
    result: ResultType

    @property
    def definition(self) -> None:
        """Definition"""

        raise NotImplementedError

    def __init__(self, parameters: ParametersType, result: ResultType) -> None:
        self.parameters = parameters
        self.result = result
