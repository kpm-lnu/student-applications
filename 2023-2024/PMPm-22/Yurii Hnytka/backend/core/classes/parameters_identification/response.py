"""Parameters Identification Response Type"""


from core.classes.common.error_response import ErrorResponse
from core.classes.parameters_identification.success_response import PISuccessResponse


type PIResponse = PISuccessResponse | ErrorResponse
