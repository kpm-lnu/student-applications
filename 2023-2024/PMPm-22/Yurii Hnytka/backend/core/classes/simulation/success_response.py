"""Simulation Success Response Class"""


from core.classes.common.success_response import SuccessResponse
from core.classes.simulation.parameters import SimulationParameters
from core.classes.simulation.result import SimulationResult
from core.definitions.simulation.success_response import SimulationSuccessResponseDefinition


class SimulationSuccessResponse(SuccessResponse[SimulationParameters, SimulationResult]):
    """Simulation Success Response"""

    @property
    def definition(self) -> SimulationSuccessResponseDefinition:
        """Definition"""

        return {
            'parameters': self.parameters.definition,
            'result': self.result.definition
        }

    def __init__(self, definition: SimulationSuccessResponseDefinition) -> None:
        super().__init__(
            SimulationParameters(definition['parameters']),
            SimulationResult(definition['result'])
        )
