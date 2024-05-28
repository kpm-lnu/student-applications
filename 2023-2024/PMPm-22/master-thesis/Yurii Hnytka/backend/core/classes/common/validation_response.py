"""Validation Response Class"""


from core.definitions.common.validation_response import ValidationResponseDefinition


class ValidationResponse:
    """Validation Response"""

    valid: bool
    message: str | None

    @property
    def definition(self) -> ValidationResponseDefinition:
        """Definition"""

        return {
            'valid': self.valid,
            'message': self.message
        }

    def __init__(self, definition: ValidationResponseDefinition) -> None:
        self.valid = definition['valid']
        self.message = definition['message']
