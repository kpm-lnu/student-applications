"""Optimal Control Middleware"""


from typing import Callable, cast
from scipy.optimize import minimize, OptimizeResult
from sympy import Symbol, Interval, oo

from core.classes.common.error_response import ErrorResponse
from core.classes.common.values import Values
from core.classes.model.equation import Equation
from core.classes.model.model import Model
from core.classes.model.variables_datatable import VariablesDatatable
from core.classes.optimal_control.adjoint_model import AdjointModel
from core.classes.optimal_control.parameters import OptimalControlParameters
from core.classes.optimal_control.response import OptimalControlResponse
from core.classes.optimal_control.success_response import OptimalControlSuccessResponse
from core.definitions.model.continuity_type import ContinuityType
from core.definitions.optimal_control.approximation_type import ApproximationType
from core.definitions.optimal_control.result import OptimalControlResultDefinition
from core.definitions.simulation.result import SimulationResultDefinition


def optimal_control(parameters: OptimalControlParameters, model: Model) -> OptimalControlResponse:
    """Optimal Control"""

    for compartment in model.compartments:
        continuity_status: dict[str, ContinuityType] = {
            **{
                str(symbol):
                compartment.equation.check_continuity(
                    symbol, Interval(0, oo), True
                )
                for symbol in model.compartments_symbols
            },
            **{
                str(symbol):
                compartment.equation.check_continuity(
                    symbol,
                    Interval(
                        parameters.intervention.lower_boundary,
                        parameters.intervention.upper_boundary
                    ),
                    True
                )
                for symbol in model.interventions_symbols
            }
        }

        if not all([
            continuity_type == ContinuityType.CONTINUOUSLY_DIFFERENTIABLE
            for continuity_type in continuity_status.values()
        ]):
            discontinuous_symbols: list[str] = [
                symbol
                for symbol, continuity_type in continuity_status.items()
                if continuity_type == ContinuityType.DISCONTINUOUS
            ]
            continuous_symbols: list[str] = [
                symbol
                for symbol, continuity_type in continuity_status.items()
                if continuity_type == ContinuityType.CONTINUOUS
            ]

            return ErrorResponse({
                'error': f'Equation of {compartment.name} is discontinuous by:\n' +
                ', '.join(discontinuous_symbols) +
                '\nand only continuous by:\n' +
                ', '.join(continuous_symbols)
            })

    if not model.population_preserved:
        return ErrorResponse({
            'error': 'Model equations do not preserve population\n\n' +
            'Apparently that is result of a program bug. Please reload the page and try again',
        })

    cost_function: Equation = Equation()

    cost_function.add_str(
        parameters.objective_function,
        model.symbols_table
    )

    cost_function.substitute(
        [
            (constant.symbol, constant.value)
            for constant in model.constants
        ]
    )

    continuity_status: dict[str, ContinuityType] = {
        **{
            str(symbol):
            cost_function.check_continuity(
                symbol, Interval(0, oo), True
            )
            for symbol in model.compartments_symbols
        },
        **{
            str(symbol):
            cost_function.check_continuity(
                symbol,
                Interval(
                    parameters.intervention.lower_boundary,
                    parameters.intervention.upper_boundary
                ),
                True
            )
            for symbol in model.interventions_symbols
        }
    }

    if not all([
        continuity_type == ContinuityType.CONTINUOUSLY_DIFFERENTIABLE
        for continuity_type in continuity_status.values()
    ]):
        discontinuous_symbols: list[str] = [
            symbol
            for symbol, continuity_type in continuity_status.items()
            if continuity_type == ContinuityType.DISCONTINUOUS
        ]
        continuous_symbols: list[str] = [
            symbol
            for symbol, continuity_type in continuity_status.items()
            if continuity_type == ContinuityType.CONTINUOUS
        ]

        return ErrorResponse({
            'error': 'Equation of cost function is discontinuous by:\n' +
            ', '.join(discontinuous_symbols) +
            '\nand only continuous by:\n' +
            ', '.join(continuous_symbols)
        })

    hamiltonian: Equation = Equation()

    hamiltonian.add(cost_function.expression)

    for compartment in model.compartments:
        hamiltonian.add(
            Symbol(f'lambda_{compartment.name}') *
            compartment.equation.expression  # type: ignore
        )

    adjoint_model: AdjointModel = AdjointModel(
        model.compartments_symbols,
        hamiltonian,
    )

    try:
        variables_datatable: VariablesDatatable = VariablesDatatable()

        variables_datatable.update_interventions({
            intervention.name: Values({
                'name': intervention.name,
                'values': [0] * (parameters.nodes_amount + 1)
            })
            for intervention in model.interventions
        })

        model.simulate(
            parameters.time / parameters.nodes_amount,
            parameters.nodes_amount,
            variables_datatable
        )

        no_control_result: SimulationResultDefinition = {
            'compartments': variables_datatable.compartments_definition
        }

        no_control_cost: float = cost_function.calculate_interval(
            parameters.time / parameters.nodes_amount,
            parameters.nodes_amount,
            variables_datatable
        )

        if parameters.intervention.approximation_type == ApproximationType.PIECEWISE_LINEAR:
            parameters.intervention.nodes_amount += 1

        optimal_result: OptimizeResult = cast(
            OptimizeResult,
            minimize(
                optimization_criteria,
                [0] * parameters.intervention.nodes_amount *
                len(model.interventions),
                args=(
                    parameters,
                    hamiltonian,
                    model,
                    adjoint_model,
                    variables_datatable,
                ),
                bounds=[
                    (
                        parameters.intervention.lower_boundary,
                        parameters.intervention.upper_boundary
                    )
                    for _ in range(parameters.intervention.nodes_amount)
                ] * len(model.interventions),
                method='L-BFGS-B',
                tol=1e-3
            )
        )

        optimal_cost: float = cost_function.calculate_interval(
            parameters.time / parameters.nodes_amount,
            parameters.nodes_amount,
            variables_datatable
        )

        result: OptimalControlResultDefinition = {
            'compartments': variables_datatable.compartments_definition,
            'interventions': [
                {
                    'name': intervention.name,
                    'values': list(
                        optimal_result.x[
                            i * parameters.intervention.nodes_amount:
                            (i + 1) * parameters.intervention.nodes_amount
                        ]
                    )
                }
                for i, intervention in enumerate(model.interventions)
            ],
            'approximatedInterventions': variables_datatable.interventions_definition,
            'noControlObjective': no_control_cost,
            'optimalObjective': optimal_cost,
        }

        return OptimalControlSuccessResponse({
            'parameters': parameters.definition,
            'result': (
                no_control_result,
                result
            )
        })

    except ValueError as error:
        return ErrorResponse({
            'error': str(error)
        })


def optimization_criteria(
    interventions_vector: list[float],
    parameters: OptimalControlParameters,
    hamiltonian: Equation,
    model: Model,
    adjoint_model: AdjointModel,
    variables_datatable: VariablesDatatable,
) -> float:
    """Optimization criteria"""

    step_size: float = parameters.time / parameters.nodes_amount

    approximation_function: Callable[[float, list[float], float], float] = (
        piecewise_constant_control_approximation
        if parameters.intervention.approximation_type == ApproximationType.PIECEWISE_CONSTANT
        else piecewise_linear_control_approximation
    )

    variables_datatable.clear()

    variables_datatable.update_interventions({
        intervention.name: Values({
            'name': intervention.name,
            'values': [
                approximation_function(
                    j * step_size,
                    interventions_vector[
                        i * parameters.intervention.nodes_amount:
                        (i + 1) * parameters.intervention.nodes_amount
                    ],
                    parameters.time
                )
                for j in range(parameters.nodes_amount + 1)
            ]
        })
        for i, intervention in enumerate(model.interventions)
    })

    model.simulate(
        step_size,
        parameters.nodes_amount,
        variables_datatable,
        True
    )

    adjoint_model.simulate(
        step_size,
        parameters.nodes_amount,
        variables_datatable
    )

    return hamiltonian.calculate_interval(
        step_size,
        parameters.nodes_amount,
        variables_datatable
    )


def piecewise_constant_control_approximation(
    t: float,
    interventions_vector: list[float],
    simulation_time: float,
) -> float:
    """Constant Control approximation"""

    if t >= simulation_time:
        return interventions_vector[-1]

    intervention_nodes_step: float = simulation_time / \
        len(interventions_vector)

    return interventions_vector[
        int(t / intervention_nodes_step)
    ]


def piecewise_linear_control_approximation(
    t: float,
    interventions_vector: list[float],
    simulation_time: float,
) -> float:
    """Linear Control approximation"""

    if t >= simulation_time:
        return interventions_vector[-1]

    intervention_nodes_step: float = simulation_time / \
        (len(interventions_vector) - 1)

    i: int = int(t / intervention_nodes_step)

    t_right: float = (i + 1) * intervention_nodes_step
    t_left: float = i * intervention_nodes_step

    return (
        (t_right - t) / (t_right - t_left) * interventions_vector[i] +
        (t - t_left) / (t_right - t_left) * interventions_vector[i + 1]
    )
