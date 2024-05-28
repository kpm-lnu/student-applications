"""Validation Request Body"""


from core.definitions.common.validation_request_body import ValidationRequestBodyDefinition


class ValidationRequestBody:
    """Validation request body"""
    expression: str
    allowed_symbols: list[str]

    @property
    def definition(self) -> ValidationRequestBodyDefinition:
        """Return definition"""

        return {
            'expression': self.expression,
            'allowedSymbols': self.allowed_symbols
        }

    def __init__(self, definition: ValidationRequestBodyDefinition) -> None:
        self.expression = definition['expression']
        self.allowed_symbols = definition['allowedSymbols']
