"""Simulation Result Class"""


from core.classes.common.values import Values
from core.definitions.simulation.result import SimulationResultDefinition


class SimulationResult:
    """Simulation Result"""

    compartments: list[Values]

    @property
    def definition(self) -> SimulationResultDefinition:
        """Definition"""

        return {
            'compartments': [
                compartment.definition
                for compartment in self.compartments
            ]
        }

    def __init__(self, definition: SimulationResultDefinition) -> None:
        self.compartments = [
            Values(compartment)
            for compartment in definition['compartments']
        ]
