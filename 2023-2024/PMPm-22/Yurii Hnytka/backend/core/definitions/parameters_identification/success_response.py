"""Parameters Identification Success Response Definition"""


from typing import TypedDict

from core.definitions.parameters_identification.parameters import PIParametersDefinition
from core.definitions.parameters_identification.result import PIResultDefinition


class PISuccessResponseDefinition(TypedDict):
    """PI Success Response Definition"""

    parameters: PIParametersDefinition
    result: PIResultDefinition
