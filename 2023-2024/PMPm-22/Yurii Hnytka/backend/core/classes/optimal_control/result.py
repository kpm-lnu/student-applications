"""Optimal Control Result Class"""


from core.classes.common.values import Values
from core.definitions.optimal_control.result import OptimalControlResultDefinition


class OptimalControlResult:
    """Optimal Control Result"""

    compartments: list[Values]
    interventions: list[Values]
    approximated_interventions: list[Values]
    no_control_objective: float
    control_objective: float

    @property
    def definition(self) -> OptimalControlResultDefinition:
        """Definition"""

        return {
            'compartments': [
                compartment.definition
                for compartment in self.compartments
            ],
            'interventions': [
                intervention.definition
                for intervention in self.interventions
            ],
            'approximatedInterventions': [
                approximated_intervention.definition
                for approximated_intervention in self.approximated_interventions
            ],
            'noControlObjective': self.no_control_objective,
            'optimalObjective': self.control_objective,
        }

    def __init__(self, definition: OptimalControlResultDefinition) -> None:
        self.compartments = [
            Values(compartment)
            for compartment in definition['compartments']
        ]
        self.interventions = [
            Values(intervention)
            for intervention in definition['interventions']
        ]
        self.approximated_interventions = [
            Values(approximated_intervention)
            for approximated_intervention in definition['approximatedInterventions']
        ]
        self.no_control_objective = definition['noControlObjective']
        self.control_objective = definition['optimalObjective']
