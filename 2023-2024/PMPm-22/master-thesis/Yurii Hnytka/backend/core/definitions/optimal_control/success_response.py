"""Optimal Control Success Response Definition"""


from typing import TypedDict

from core.definitions.optimal_control.parameters import OptimalControlParametersDefinition
from core.definitions.optimal_control.result import OptimalControlResultDefinition
from core.definitions.simulation.result import SimulationResultDefinition


class OptimalControlSuccessResponseDefinition(TypedDict):
    """Optimal Control Success Response Definition"""

    parameters: OptimalControlParametersDefinition
    result: tuple[SimulationResultDefinition, OptimalControlResultDefinition]
