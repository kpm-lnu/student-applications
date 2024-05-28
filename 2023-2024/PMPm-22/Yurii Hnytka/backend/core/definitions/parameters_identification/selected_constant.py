"""Selected Constant Definition"""


from typing import TypedDict


class SelectedConstantDefinition(TypedDict):
    """Selected Constant Definition"""

    id: str
    name: str
    upperBoundary: float
    value: float
    lowerBoundary: float
