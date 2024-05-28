"""Intervention Parameters Class"""


from core.definitions.optimal_control.approximation_type import ApproximationType
from core.definitions.optimal_control.intervention_parameters import InterventionParametersDefinition


class InterventionParameters:
    """Intervention Parameters"""

    nodes_amount: int
    approximation_type: ApproximationType
    lower_boundary: float
    upper_boundary: float

    @property
    def definition(self) -> InterventionParametersDefinition:
        """Definition"""

        return {
            'nodesAmount': self.nodes_amount,
            'approximationType': self.approximation_type,
            'lowerBoundary': self.lower_boundary,
            'upperBoundary': self.upper_boundary,
        }

    def __init__(self, definition: InterventionParametersDefinition) -> None:
        self.nodes_amount = definition['nodesAmount']
        self.approximation_type = ApproximationType(
            definition['approximationType']
        )
        self.lower_boundary = definition['lowerBoundary']
        self.upper_boundary = definition['upperBoundary']
