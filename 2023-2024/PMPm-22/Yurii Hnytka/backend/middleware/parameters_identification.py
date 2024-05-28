"""Parameters Identification Middleware"""


from typing import cast
from scipy.optimize import minimize, OptimizeResult
from sympy import Interval, oo

from core.classes.common.error_response import ErrorResponse
from core.classes.common.values import Values
from core.classes.model.model import Model
from core.classes.model.variables_datatable import VariablesDatatable
from core.classes.parameters_identification.parameters import PIParameters
from core.classes.parameters_identification.response import PIResponse
from core.classes.parameters_identification.success_response import PISuccessResponse
from core.definitions.model.continuity_type import ContinuityType
from core.definitions.parameters_identification.result import PIResultDefinition


def parameters_identification(parameters: PIParameters, model: Model) -> PIResponse:
    """Identify the parameters"""

    model.symbols_table.update({
        constant.name: constant.symbol
        for constant in parameters.selected_constants
    })

    for compartment in model.compartments:
        continuity_status: dict[str, ContinuityType] = {
            **{
                str(symbol):
                compartment.equation.check_continuity(
                    symbol, Interval(0, oo)
                )
                for symbol in model.compartments_symbols
            },
            **{
                constant.name:
                compartment.equation.check_continuity(
                    constant.symbol,
                    Interval(
                        constant.lower_boundary,
                        constant.upper_boundary
                    )
                )
                for constant in parameters.selected_constants
            }
        }

        if not all([
            continuity_type == ContinuityType.CONTINUOUS
            for continuity_type in continuity_status.values()
        ]):
            discontinuous_symbols: list[str] = [
                symbol
                for symbol, continuity_type in continuity_status.items()
                if continuity_type == ContinuityType.DISCONTINUOUS
            ]

            return ErrorResponse({
                'error': f'Equation of {compartment.name} is discontinuous by:\n' +
                ', '.join(discontinuous_symbols)
            })

    if not model.population_preserved:
        return ErrorResponse({
            'error': 'Model equations do not preserve population\n\n' +
            'Apparently that is result of a program bug. Please reload the page and try again',
        })

    try:
        variables_datatable: VariablesDatatable = VariablesDatatable()

        minimize_result: OptimizeResult = cast(
            OptimizeResult,
            minimize(
                optimization_criteria,
                [
                    constant.value
                    for constant in parameters.selected_constants
                ],
                args=(
                    parameters,
                    model,
                    variables_datatable
                ),
                bounds=[
                    (
                        constant.lower_boundary,
                        constant.upper_boundary
                    )
                    for constant in parameters.selected_constants
                ],
                method='L-BFGS-B',
                tol=1e-3
            )
        )

        result: PIResultDefinition = {
            'approximation': variables_datatable.compartments_definition,
            'constants': [
                {
                    'id': constant.id,
                    'name': constant.name,
                    'value': minimize_result.x[i],
                }
                for i, constant in enumerate(parameters.selected_constants)
            ],
        }

        return PISuccessResponse({
            'parameters': parameters.definition,
            'result': result,
        })

    except ValueError as error:
        return ErrorResponse({
            'error': str(error)
        })


def optimization_criteria(
    constants_vector: list[float],
    parameters: PIParameters,
    model: Model,
    variables_datatable: VariablesDatatable,
) -> float:
    """Optimization criteria"""

    nodes_amount: int = len(parameters.data[0].values) - 1

    variables_datatable.update_constants({
        constant.name: Values({
            'name': constant.name,
            'values': [constants_vector[i]] * nodes_amount
        })
        for i, constant in enumerate(parameters.selected_constants)
    })

    variables_datatable.update_interventions({
        intervention.name: Values({
            'name': intervention.name,
            'values': next(
                (
                    values.values
                    for values in parameters.data
                    if values.name == values.name
                ),
                []
            )
        })
        for intervention in model.interventions
    })

    model.simulate(
        parameters.time_step,
        nodes_amount,
        variables_datatable,
        True
    )

    return calculate_objective(
        parameters.data,
        variables_datatable.compartments,
    )


def calculate_objective(
    data: list[Values],
    simulation_data: dict[str, Values],
) -> float:
    """Cost function"""

    cost: float = 0

    for values in data:
        compartment_values: Values = simulation_data[values.name]

        for i, value in enumerate(values.values):
            cost += (value - compartment_values.values[i]) ** 2

    return cost
