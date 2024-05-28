"""Simulation Response Type"""


from core.classes.common.error_response import ErrorResponse
from core.classes.simulation.success_response import SimulationSuccessResponse


type SimulationResponse = SimulationSuccessResponse | ErrorResponse
