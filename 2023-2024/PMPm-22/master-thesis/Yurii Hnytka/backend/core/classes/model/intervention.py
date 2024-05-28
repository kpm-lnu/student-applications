"""Intervention Class"""


from sympy import Symbol

from core.definitions.model.intervention import InterventionDefinition


class Intervention:
    """Intervention"""

    id: str
    name: str

    symbol: Symbol

    @property
    def definition(self) -> InterventionDefinition:
        """Definition"""

        return {
            'id': self.id,
            'name': self.name
        }

    def __init__(self, definition: InterventionDefinition) -> None:
        self.id = definition['id']
        self.name = definition['name']

        self.symbol = Symbol(self.name)
