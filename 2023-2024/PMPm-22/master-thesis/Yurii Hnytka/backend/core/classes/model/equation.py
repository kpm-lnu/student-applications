"""Equation Class"""


from typing import Callable, cast
from sympy import Expr, FiniteSet, Interval, Symbol, diff, lambdify, sympify
from sympy.calculus.util import continuous_domain

from core.classes.model.variables_datatable import VariablesDatatable
from core.definitions.model.symbols_table import SymbolsTable
from core.definitions.model.continuity_type import ContinuityType


class Equation():
    """Equation"""

    expression: Expr
    function: Callable[..., float]

    @property
    def variables(self) -> list[Symbol]:
        """Variables"""

        return cast(list[Symbol], list(self.expression.free_symbols))

    def __init__(self) -> None:
        self.expression = cast(Expr, 0)
        self.function = lambda *args: 0

    def calculate(self, values: list[float]) -> float:
        """Calculate"""

        return self.function(*values)

    def calculate_interval(
        self,
        step_size: float,
        nodes_amount: int,
        variables_datatable: VariablesDatatable
    ) -> float:
        """Calculate interval"""

        value: float = 0

        for i in range(nodes_amount + 1):
            values: list[float] = [
                variables_datatable[str(variable)][i]
                for variable in self.variables
            ]

            value += self.calculate(values) * step_size

        return value

    def check_continuity(
        self,
        symbol: Symbol,
        interval: Interval | FiniteSet,
        derivative: bool = False
    ) -> ContinuityType:
        """Check continuity"""

        if not continuous_domain(self.expression, symbol, interval) == interval:
            return ContinuityType.DISCONTINUOUS

        if derivative:
            if not continuous_domain(
                diff(self.expression, symbol),
                symbol,
                interval
            ) == interval:
                return ContinuityType.CONTINUOUS

        return (
            ContinuityType.CONTINUOUSLY_DIFFERENTIABLE
            if derivative else
            ContinuityType.CONTINUOUS
        )

    def add(self, equation: Expr) -> None:
        """Add equation"""

        self.expression += equation  # type: ignore

        self.__update_function()

    def add_str(self, equation: str, symbols_table: SymbolsTable) -> None:
        """Add equation"""

        self.expression += sympify(
            equation.replace('^', '**'),
            symbols_table
        )

        self.__update_function()

    def subtract(self, equation: Expr) -> None:
        """Subtract equation"""

        self.expression -= equation  # type: ignore

        self.__update_function()

    def subtract_str(self, equation: str, symbols_table: SymbolsTable) -> None:
        """Subtract equation"""

        self.expression -= sympify(
            equation.replace('^', '**'),
            symbols_table
        )

        self.__update_function()

    def substitute(self, constants: list[tuple[Symbol, float]]) -> None:
        """Substitute constants"""

        self.expression = cast(Expr, self.expression.subs(constants))

        self.__update_function()

    def __update_function(self) -> None:
        """Update Function"""

        self.function = lambdify(self.variables, self.expression)
