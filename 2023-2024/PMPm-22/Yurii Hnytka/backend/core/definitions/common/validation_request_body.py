"""Validation Request Body Definition"""


from typing import TypedDict


class ValidationRequestBodyDefinition(TypedDict):
    """Validation Request Body Definition"""

    expression: str
    allowedSymbols: list[str]
