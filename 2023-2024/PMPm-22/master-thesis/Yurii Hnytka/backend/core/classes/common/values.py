"""Values Class"""


from core.definitions.common.values import ValuesDefinition


class Values:
    """Values"""

    name: str
    values: list[float]

    @property
    def definition(self) -> ValuesDefinition:
        """Definition"""

        return {
            'name': self.name,
            'values': self.values
        }

    def __init__(self, definition: ValuesDefinition) -> None:
        self.name = definition['name']
        self.values = definition['values']
