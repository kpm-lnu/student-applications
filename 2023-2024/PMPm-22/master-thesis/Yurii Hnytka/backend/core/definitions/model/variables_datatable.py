"""Variables Datatable Definition"""


from typing import TypedDict

from core.definitions.common.values import ValuesDefinition


class VariablesDatatableDefinition(TypedDict):
    """Variables Datatable Definition"""

    compartments: dict[str, ValuesDefinition]
    constants: dict[str, ValuesDefinition]
    interventions: dict[str, ValuesDefinition]
    lambdas: dict[str, ValuesDefinition]
