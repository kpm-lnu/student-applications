"""Parameter Identification Result Definition"""


from typing import TypedDict

from core.definitions.common.values import ValuesDefinition
from core.definitions.parameters_identification.identified_constant import IdentifiedConstantDefinition


class PIResultDefinition(TypedDict):
    """PI Result Definition"""

    constants: list[IdentifiedConstantDefinition]
    approximation: list[ValuesDefinition]
