"""Optimal Control Request Body Definition"""


from core.definitions.common.request_body import RequestBodyDefinition
from core.definitions.optimal_control.parameters import OptimalControlParametersDefinition


OptimalControlRequestBodyDefinition = RequestBodyDefinition[OptimalControlParametersDefinition]
