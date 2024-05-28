"""Flow Definition"""


from typing import TypedDict


class FlowDefinition(TypedDict):
    """Flow Definition"""

    id: str
    source: str
    target: str
    equation: str
