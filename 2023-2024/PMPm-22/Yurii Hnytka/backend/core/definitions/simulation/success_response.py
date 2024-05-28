"""Simulation Success Response Definition"""


from typing import TypedDict

from core.definitions.simulation.parameters import SimulationParametersDefinition
from core.definitions.simulation.result import SimulationResultDefinition


class SimulationSuccessResponseDefinition(TypedDict):
    """Simulation Success Response Definition"""

    parameters: SimulationParametersDefinition
    result: SimulationResultDefinition
