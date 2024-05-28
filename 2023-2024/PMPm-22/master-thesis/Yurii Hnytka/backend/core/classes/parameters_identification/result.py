"""Parameter Identification Result Class"""


from core.classes.common.values import Values
from core.classes.parameters_identification.identified_constant import IdentifiedConstant
from core.definitions.parameters_identification.result import PIResultDefinition


class PIResult:
    """PI Result"""

    constants: list[IdentifiedConstant]
    approximation: list[Values]

    @property
    def definition(self) -> PIResultDefinition:
        """Definition"""

        return {
            'constants': [
                constant.definition
                for constant in self.constants
            ],
            'approximation': [
                approximation.definition
                for approximation in self.approximation
            ]
        }

    def __init__(self, definition: PIResultDefinition) -> None:
        self.constants = [
            IdentifiedConstant(constant)
            for constant in definition['constants']
        ]
        self.approximation = [
            Values(approximation)
            for approximation in definition['approximation']
        ]
