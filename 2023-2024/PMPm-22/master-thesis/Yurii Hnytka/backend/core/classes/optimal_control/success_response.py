"""Optimal Control Success Response Class"""


from core.classes.common.success_response import SuccessResponse
from core.classes.optimal_control.parameters import OptimalControlParameters
from core.classes.optimal_control.result import OptimalControlResult
from core.classes.simulation.result import SimulationResult
from core.definitions.optimal_control.success_response import OptimalControlSuccessResponseDefinition


class OptimalControlSuccessResponse(
    SuccessResponse[
        OptimalControlParameters,
        tuple[SimulationResult, OptimalControlResult]
    ]
):
    """Optimal Control Success Response"""

    @property
    def definition(self) -> OptimalControlSuccessResponseDefinition:
        """Definition"""

        return {
            'parameters': self.parameters.definition,
            'result': (
                self.result[0].definition,
                self.result[1].definition
            )
        }

    def __init__(self, definition: OptimalControlSuccessResponseDefinition) -> None:
        super().__init__(
            OptimalControlParameters(
                definition['parameters']
            ),
            (
                SimulationResult(
                    definition['result'][0]
                ),
                OptimalControlResult(
                    definition['result'][1]
                )
            )
        )
