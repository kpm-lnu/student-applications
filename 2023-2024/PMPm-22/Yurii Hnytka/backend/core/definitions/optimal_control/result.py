"""Optimal Control Result Definition"""


from typing import TypedDict

from core.definitions.common.values import ValuesDefinition


class OptimalControlResultDefinition(TypedDict):
    """Optimal Control Result Definition"""

    compartments: list[ValuesDefinition]
    interventions: list[ValuesDefinition]
    approximatedInterventions: list[ValuesDefinition]
    noControlObjective: float
    optimalObjective: float
