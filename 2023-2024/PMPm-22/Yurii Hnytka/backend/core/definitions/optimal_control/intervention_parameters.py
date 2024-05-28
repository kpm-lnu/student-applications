"""Intervention Parameters Definition"""


from typing import TypedDict

from core.definitions.optimal_control.approximation_type import ApproximationType


class InterventionParametersDefinition(TypedDict):
    """Intervention Parameters Definition"""

    nodesAmount: int
    approximationType: ApproximationType
    lowerBoundary: float
    upperBoundary: float
