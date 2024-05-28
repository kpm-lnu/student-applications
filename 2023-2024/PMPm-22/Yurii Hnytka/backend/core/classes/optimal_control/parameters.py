"""Optimal Control Parameters Class"""


from core.classes.optimal_control.intervention_parameters import InterventionParameters
from core.definitions.optimal_control.parameters import OptimalControlParametersDefinition


class OptimalControlParameters:
    """Optimal Control Parameters"""

    time: float
    nodes_amount: int
    objective_function: str
    intervention: InterventionParameters

    @property
    def definition(self) -> OptimalControlParametersDefinition:
        """Definition"""

        return {
            'time': self.time,
            'nodesAmount': self.nodes_amount,
            'objectiveFunction': self.objective_function,
            'intervention': self.intervention.definition
        }

    def __init__(self, definition: OptimalControlParametersDefinition) -> None:
        self.time = definition['time']
        self.nodes_amount = definition['nodesAmount']
        self.objective_function = definition['objectiveFunction']
        self.intervention = InterventionParameters(
            definition['intervention']
        )
