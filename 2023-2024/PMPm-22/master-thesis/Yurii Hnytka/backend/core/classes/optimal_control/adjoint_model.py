"""Adjoint Model Class"""


from sympy import Symbol

from core.classes.common.values import Values
from core.classes.model.equation import Equation
from core.classes.model.variables_datatable import VariablesDatatable
from core.classes.optimal_control.lambda_derivative import LambdaDerivative


class AdjointModel:
    """Adjoint Model"""

    lambdas: list[LambdaDerivative]

    def __init__(self, compartments: list[Symbol], hamiltonian: Equation) -> None:
        self.lambdas = [
            LambdaDerivative(
                f'lambda_{compartment}',
                compartment,
                hamiltonian
            )
            for compartment in compartments
        ]

    def simulate(
        self,
        step_size: float,
        nodes_amount: int,
        variables_datatable: VariablesDatatable
    ) -> None:
        """Simulate"""

        variables_datatable.update_lambdas({
            current_lambda.name: Values({
                'name': current_lambda.name,
                'values': [0] * (nodes_amount + 1)
            })
            for current_lambda in self.lambdas
        })

        for i in range(nodes_amount - 1, -1, -1):
            for current_lambda in self.lambdas:
                values: list[float] = [
                    variables_datatable[str(variable)][i + 1]
                    for variable in current_lambda.equation.variables
                ]

                variables_datatable[current_lambda.name][i] = (
                    variables_datatable[current_lambda.name][i + 1] -
                    current_lambda.equation.calculate(values) * step_size
                )
