"""Optimal Control Response Type"""


from core.classes.common.error_response import ErrorResponse
from core.classes.optimal_control.success_response import OptimalControlSuccessResponse


type OptimalControlResponse = OptimalControlSuccessResponse | ErrorResponse
