"""Selected Constant Class"""


from sympy import Symbol

from core.definitions.parameters_identification.selected_constant import SelectedConstantDefinition


class SelectedConstant:
    """Selected constant"""

    id: str
    name: str
    upper_boundary: float
    value: float
    lower_boundary: float

    symbol: Symbol

    @property
    def definition(self) -> SelectedConstantDefinition:
        """Definition"""

        return {
            'id': self.id,
            'name': self.name,
            'upperBoundary': self.upper_boundary,
            'value': self.value,
            'lowerBoundary': self.lower_boundary
        }

    def __init__(self, definition: SelectedConstantDefinition) -> None:
        self.id = definition['id']
        self.name = definition['name']
        self.upper_boundary = definition['upperBoundary']
        self.value = definition['value']
        self.lower_boundary = definition['lowerBoundary']

        self.symbol = Symbol(self.name)
