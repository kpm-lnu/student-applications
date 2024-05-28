"""Model Definition"""


from typing import TypedDict

from core.definitions.model.compartment import CompartmentDefinition
from core.definitions.model.constant import ConstantDefinition
from core.definitions.model.intervention import InterventionDefinition
from core.definitions.model.flow import FlowDefinition


class ModelDefinition(TypedDict):
    """Model Definition"""

    compartments: list[CompartmentDefinition]
    constants: list[ConstantDefinition]
    interventions: list[InterventionDefinition]
    flows: list[FlowDefinition]
