"""Optimal Control Parameters Definition"""


from typing import TypedDict

from core.definitions.optimal_control.intervention_parameters import InterventionParametersDefinition


class OptimalControlParametersDefinition(TypedDict):
    """Optimal Control Parameters Definition"""

    time: float
    nodesAmount: int
    objectiveFunction: str
    intervention: InterventionParametersDefinition
