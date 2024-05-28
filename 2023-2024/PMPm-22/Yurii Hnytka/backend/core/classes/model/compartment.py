"""Compartment Class"""


from sympy import Symbol

from core.classes.model.equation import Equation
from core.definitions.model.compartment import CompartmentDefinition
from core.definitions.model.symbols_table import SymbolsTable


class Compartment:
    """Compartment"""

    id: str
    name: str
    value: float

    symbol: Symbol
    equation: Equation

    @property
    def definition(self) -> CompartmentDefinition:
        """Definition"""

        return {
            'id': self.id,
            'name': self.name,
            'value': self.value,
        }

    @property
    def variables(self) -> list[Symbol]:
        """Variables"""

        return self.equation.variables

    def __init__(self, definition: CompartmentDefinition) -> None:
        self.id = definition['id']
        self.name = definition['name']
        self.value = definition['value']

        self.symbol = Symbol(self.name)
        self.equation = Equation()

    def add_inflow(self, inflow: str, symbols_table: SymbolsTable) -> None:
        """Add inflow"""

        self.equation.add_str(inflow, symbols_table)

    def add_outflow(self, outflow: str, symbols_table: SymbolsTable) -> None:
        """Add outflow"""

        self.equation.subtract_str(outflow, symbols_table)
