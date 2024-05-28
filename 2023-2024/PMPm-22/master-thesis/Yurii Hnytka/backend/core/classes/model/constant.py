"""Constant Class"""


from sympy import Symbol

from core.definitions.model.constant import ConstantDefinition


class Constant:
    """Constant"""

    id: str
    name: str
    value: float

    symbol: Symbol

    @property
    def definition(self) -> ConstantDefinition:
        """Definition"""

        return {
            'id': self.id,
            'name': self.name,
            'value': self.value
        }

    def __init__(self, definition: ConstantDefinition) -> None:
        self.id = definition['id']
        self.name = definition['name']
        self.value = definition['value']

        self.symbol = Symbol(self.name)
