"""Parameter Identification Request Body Definition"""


from core.definitions.common.request_body import RequestBodyDefinition
from core.definitions.parameters_identification.parameters import PIParametersDefinition


PIRequestBodyDefinition = RequestBodyDefinition[PIParametersDefinition]
