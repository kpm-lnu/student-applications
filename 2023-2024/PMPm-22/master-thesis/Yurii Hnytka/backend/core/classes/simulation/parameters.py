"""Simulation Parameters Class"""


from core.definitions.simulation.parameters import SimulationParametersDefinition


class SimulationParameters:
    """Simulation Parameters"""

    time: float
    nodes_amount: int

    @property
    def definition(self) -> SimulationParametersDefinition:
        """Definition"""

        return {
            'time': self.time,
            'nodesAmount': self.nodes_amount,
        }

    def __init__(self, definition: SimulationParametersDefinition) -> None:
        self.time = definition['time']
        self.nodes_amount = definition['nodesAmount']
