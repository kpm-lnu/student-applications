"""Optimal Control Request Body Class"""


from core.classes.common.request_body import RequestBody
from core.classes.model.model import Model
from core.classes.optimal_control.parameters import OptimalControlParameters
from core.definitions.optimal_control.request_body import OptimalControlRequestBodyDefinition


class OptimalControlRequestBody(RequestBody[OptimalControlParameters]):
    """Optimal Control Request Body"""

    @property
    def definition(self) -> OptimalControlRequestBodyDefinition:
        """Definition"""

        return {
            'parameters': self.parameters.definition,
            'model': self.model.definition
        }

    def __init__(self, definition: OptimalControlRequestBodyDefinition) -> None:
        super().__init__(
            OptimalControlParameters(definition['parameters']),
            Model(definition['model'])
        )
