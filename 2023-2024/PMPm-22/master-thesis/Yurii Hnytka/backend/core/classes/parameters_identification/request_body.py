"""Parameter Identification Request Body Class"""


from core.classes.common.request_body import RequestBody
from core.classes.model.model import Model
from core.classes.parameters_identification.parameters import PIParameters
from core.definitions.parameters_identification.request_body import PIRequestBodyDefinition


class PIRequestBody(RequestBody[PIParameters]):
    """PI Request Body"""

    @property
    def definition(self) -> PIRequestBodyDefinition:
        """Definition"""

        return {
            'parameters': self.parameters.definition,
            'model': self.model.definition
        }

    def __init__(self, definition: PIRequestBodyDefinition) -> None:
        super().__init__(
            PIParameters(
                definition['parameters']),
            Model(definition['model'])
        )
