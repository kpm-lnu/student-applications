"""Validation Response Definition"""


from typing import TypedDict


class ValidationResponseDefinition(TypedDict):
    """Validation Response Definition"""

    valid: bool
    message: str | None
