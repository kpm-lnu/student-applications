"""Error Response Class"""


from core.definitions.common.error_response import ErrorResponseDefinition


class ErrorResponse:
    """Error Response"""

    error: str

    @property
    def definition(self) -> ErrorResponseDefinition:
        """Definition"""

        return {
            'error': self.error,
        }

    def __init__(self, definition: ErrorResponseDefinition) -> None:
        self.error = definition['error']
