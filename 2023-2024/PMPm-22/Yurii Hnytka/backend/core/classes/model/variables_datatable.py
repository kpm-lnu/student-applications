"""Variables Datatable"""


from core.classes.common.values import Values
from core.definitions.common.values import ValuesDefinition
from core.definitions.model.variables_datatable import VariablesDatatableDefinition


class VariablesDatatable:
    """Variables Datatable"""

    compartments: dict[str, Values]
    constants: dict[str, Values]
    interventions: dict[str, Values]
    lambdas: dict[str, Values]

    @property
    def compartments_definition(self) -> list[ValuesDefinition]:
        """Compartments Definition"""

        return [compartment.definition for compartment in self.compartments.values()]

    @property
    def constants_definition(self) -> list[ValuesDefinition]:
        """Constants Definition"""

        return [constant.definition for constant in self.constants.values()]

    @property
    def interventions_definition(self) -> list[ValuesDefinition]:
        """Interventions Definition"""

        return [intervention.definition for intervention in self.interventions.values()]

    @property
    def lambdas_definition(self) -> list[ValuesDefinition]:
        """Lambdas Definition"""

        return [lambdas.definition for lambdas in self.lambdas.values()]

    @property
    def definition(self) -> VariablesDatatableDefinition:
        """Definition"""

        return {
            'compartments': {
                name: compartment.definition
                for name, compartment in self.compartments.items()
            },
            'constants': {
                name: constant.definition
                for name, constant in self.constants.items()
            },
            'interventions': {
                name: intervention.definition
                for name, intervention in self.interventions.items()
            },
            'lambdas': {
                name: current_lambda.definition
                for name, current_lambda in self.lambdas.items()
            }
        }

    def __init__(self, definition: VariablesDatatableDefinition | None = None) -> None:
        if definition:
            self.compartments = {
                name: Values(values)
                for name, values in definition['compartments'].items()
            }
            self.constants = {
                name: Values(values)
                for name, values in definition['constants'].items()
            }
            self.interventions = {
                name: Values(values)
                for name, values in definition['interventions'].items()
            }
            self.lambdas = {
                name: Values(values)
                for name, values in definition['lambdas'].items()
            }

            return

        self.compartments = {}
        self.constants = {}
        self.interventions = {}
        self.lambdas = {}

    def __getitem__(self, key: str) -> list[float]:
        if key in self.compartments:
            return self.compartments[key].values

        if key in self.constants:
            return self.constants[key].values

        if key in self.interventions:
            return self.interventions[key].values

        if key in self.lambdas:
            return self.lambdas[key].values

        raise KeyError(f'Key "{key}" not found in Datatable')

    def __setitem__(self, key: str, value: list[float]) -> None:
        if key in self.compartments:
            self.compartments[key].values = value

            return

        if key in self.constants:
            self.constants[key].values = value

            return

        if key in self.interventions:
            self.interventions[key].values = value

            return

        if key in self.lambdas:
            self.lambdas[key].values = value

            return

        raise KeyError(f'Key "{key}" not found in Datatable')

    def __delitem__(self, key: str) -> None:
        if key in self.compartments:
            del self.compartments[key]

            return

        if key in self.constants:
            del self.constants[key]

            return

        if key in self.interventions:
            del self.interventions[key]

            return

        if key in self.lambdas:
            del self.lambdas[key]

            return

        raise KeyError(f'Key "{key}" not found in Datatable')

    def update_compartments(self, compartments: dict[str, Values]) -> None:
        """Update compartments"""

        self.compartments.update(compartments)

    def update_constants(self, constants: dict[str, Values]) -> None:
        """Update constants"""

        self.constants.update(constants)

    def update_interventions(self, interventions: dict[str, Values]) -> None:
        """Update interventions"""

        self.interventions.update(interventions)

    def update_lambdas(self, lambdas: dict[str, Values]) -> None:
        """Update lambdas"""

        self.lambdas.update(lambdas)

    def clear(self) -> None:
        """Clear datatable"""

        self.compartments.clear()
        self.constants.clear()
        self.interventions.clear()
        self.lambdas.clear()
