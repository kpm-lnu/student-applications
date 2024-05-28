"""Simulation Result Definition"""


from typing import TypedDict

from core.definitions.common.values import ValuesDefinition


class SimulationResultDefinition(TypedDict):
    """Simulation Result Definition"""

    compartments: list[ValuesDefinition]
