"""Flow Class"""


from core.definitions.model.flow import FlowDefinition


class Flow:
    """Flow"""

    id: str
    source: str
    target: str
    equation: str

    @property
    def definition(self) -> FlowDefinition:
        """Definition"""

        return {
            'id': self.id,
            'source': self.source,
            'target': self.target,
            'equation': self.equation
        }

    def __init__(self, definition: FlowDefinition) -> None:
        self.id = definition['id']
        self.source = definition['source']
        self.target = definition['target']
        self.equation = definition['equation']
