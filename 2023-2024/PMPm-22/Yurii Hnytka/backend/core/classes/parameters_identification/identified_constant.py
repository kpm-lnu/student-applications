"""Identified Constant Class"""


from core.definitions.parameters_identification.identified_constant import IdentifiedConstantDefinition


class IdentifiedConstant:
    """Identified constant"""

    id: str
    name: str
    value: float

    @property
    def definition(self) -> IdentifiedConstantDefinition:
        """Definition"""

        return {
            'id': self.id,
            'name': self.name,
            'value': self.value
        }

    def __init__(self, definition: IdentifiedConstantDefinition) -> None:
        self.id = definition['id']
        self.name = definition['name']
        self.value = definition['value']
