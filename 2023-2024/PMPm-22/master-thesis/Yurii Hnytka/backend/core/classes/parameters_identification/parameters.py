"""Parameters Identification Parameters Class"""


from core.classes.common.values import Values
from core.classes.parameters_identification.selected_constant import SelectedConstant
from core.definitions.parameters_identification.parameters import PIParametersDefinition


class PIParameters:
    """PI Parameters"""

    time_step: float
    selected_constants: list[SelectedConstant]
    data: list[Values]

    @property
    def definition(self) -> PIParametersDefinition:
        """Definition"""

        return {
            'timeStep': self.time_step,
            'selectedConstants': [
                selected_constant.definition
                for selected_constant in self.selected_constants
            ],
            'data': [
                data.definition
                for data in self.data
            ]
        }

    def __init__(self, definition: PIParametersDefinition) -> None:
        self.time_step = definition['timeStep']
        self.selected_constants = [
            SelectedConstant(selected_constant)
            for selected_constant in definition['selectedConstants']
        ]
        self.data = [
            Values(data)
            for data in definition['data']
        ]
