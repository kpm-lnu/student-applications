"""Simulation Request Body Definition"""


from core.definitions.common.request_body import RequestBodyDefinition
from core.definitions.simulation.parameters import SimulationParametersDefinition


SimulationRequestBodyDefinition = RequestBodyDefinition[SimulationParametersDefinition]
