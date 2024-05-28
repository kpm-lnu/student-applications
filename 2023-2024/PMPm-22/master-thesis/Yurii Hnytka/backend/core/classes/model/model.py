"""Model Class"""


from typing import cast
from sympy import Expr, Symbol, nsimplify

from core.classes.common.values import Values
from core.classes.model.compartment import Compartment
from core.classes.model.constant import Constant
from core.classes.model.flow import Flow
from core.classes.model.intervention import Intervention
from core.definitions.model.symbols_table import SymbolsTable
from core.definitions.model.model import ModelDefinition
from core.classes.model.variables_datatable import VariablesDatatable


class Model:
    """Model"""

    compartments: list[Compartment]
    constants: list[Constant]
    interventions: list[Intervention]
    flows: list[Flow]

    symbols_table: SymbolsTable

    @property
    def definition(self) -> ModelDefinition:
        """Definition"""

        return {
            'compartments': [
                compartment.definition
                for compartment in self.compartments
            ],
            'constants': [
                constant.definition
                for constant in self.constants
            ],
            'interventions': [
                intervention.definition
                for intervention in self.interventions
            ],
            'flows': [
                flow.definition
                for flow in self.flows
            ]
        }

    @property
    def compartments_symbols(self) -> list[Symbol]:
        """Compartments symbols"""

        return [
            compartment.symbol
            for compartment in self.compartments
        ]

    @property
    def constants_symbols(self) -> list[Symbol]:
        """Constants symbols"""

        return [
            constant.symbol
            for constant in self.constants
        ]

    @property
    def interventions_symbols(self) -> list[Symbol]:
        """Interventions symbols"""

        return [
            intervention.symbol
            for intervention in self.interventions
        ]

    @property
    def population_preserved(self) -> bool:
        """Is population preserved"""

        accumulator: Expr = cast(Expr, 0)

        for equation in [
            compartment.equation.expression
            for compartment in self.compartments
        ]:
            accumulator += equation  # type: ignore

        if nsimplify(accumulator, tolerance=1e-10) != 0:
            return False

        return True

    def __init__(self, definition: ModelDefinition) -> None:
        self.compartments = []
        self.constants = []
        self.interventions = []
        self.flows = []

        self.symbols_table = {}

        for compartment in definition['compartments']:
            self.symbols_table[compartment['name']] = Symbol(
                compartment['name']
            )

            self.compartments.append(Compartment(compartment))

        for constant in definition['constants']:
            self.symbols_table[constant['name']] = Symbol(
                constant['name']
            )

            self.constants.append(Constant(constant))

        for intervention in definition['interventions']:
            self.symbols_table[intervention['name']] = Symbol(
                intervention['name']
            )

            self.interventions.append(Intervention(intervention))

        for flow in definition['flows']:
            self.flows.append(Flow(flow))

            source: Compartment = next(
                compartment
                for compartment in self.compartments
                if compartment.id == flow['source']
            )
            target: Compartment = next(
                compartment
                for compartment in self.compartments
                if compartment.id == flow['target']
            )

            source.add_outflow(flow['equation'], self.symbols_table)
            target.add_inflow(flow['equation'], self.symbols_table)

        for compartment in self.compartments:
            compartment.equation.substitute(
                [
                    (constant.symbol, constant.value)
                    for constant in self.constants
                ]
            )

    def simulate(
        self,
        step_size: float,
        nodes_amount: int,
        variables_datatable: VariablesDatatable,
        ignore_negative_values: bool = False
    ) -> None:
        """Simulate model"""

        variables_datatable.update_compartments({
            compartment.name: Values({
                'name': compartment.name,
                'values': [compartment.value] + [0] * nodes_amount
            })
            for compartment in self.compartments
        })

        for i in range(nodes_amount):
            for compartment in self.compartments:
                values: list[float] = [
                    variables_datatable[str(variable)][i]
                    for variable in compartment.variables
                ]

                variables_datatable[compartment.name][i + 1] = (
                    variables_datatable[compartment.name][i] +
                    compartment.equation.calculate(values) * step_size
                )

                if variables_datatable[compartment.name][i + 1] < 0 and not ignore_negative_values:
                    raise ValueError(
                        f'Negative value for {
                            compartment.name
                        } at time {i + 1}'
                    )
