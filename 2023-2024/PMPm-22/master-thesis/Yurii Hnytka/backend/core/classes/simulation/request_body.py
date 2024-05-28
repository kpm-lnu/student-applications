"""Simulation Request Body Class"""


from core.classes.common.request_body import RequestBody
from core.classes.model.model import Model
from core.classes.simulation.parameters import SimulationParameters
from core.definitions.simulation.request_body import SimulationRequestBodyDefinition


class SimulationRequestBody(RequestBody[SimulationParameters]):
    """Simulation Request Body"""

    @property
    def definition(self) -> SimulationRequestBodyDefinition:
        """Definition"""

        return {
            'parameters': self.parameters.definition,
            'model': self.model.definition
        }

    def __init__(self, definition: SimulationRequestBodyDefinition) -> None:
        super().__init__(
            SimulationParameters(definition['parameters']),
            Model(definition['model'])
        )
