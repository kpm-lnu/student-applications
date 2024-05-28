"""Request Body Definition"""


from typing import Generic, TypeVar, TypedDict

from core.definitions.model.model import ModelDefinition


# pylint: disable=invalid-name
ParametersType = TypeVar('ParametersType')
# pylint: enable=invalid-name


class RequestBodyDefinition(TypedDict, Generic[ParametersType]):
    """Request Body Definition"""

    parameters: ParametersType
    model: ModelDefinition
