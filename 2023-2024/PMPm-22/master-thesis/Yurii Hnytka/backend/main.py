"""Main Module"""


from flask import Flask, Response, request, jsonify

from core.classes.common.validation_request_body import ValidationRequestBody
from core.classes.common.validation_response import ValidationResponse
from core.classes.optimal_control.request_body import OptimalControlRequestBody
from core.classes.optimal_control.response import OptimalControlResponse
from core.classes.parameters_identification.request_body import PIRequestBody
from core.classes.parameters_identification.response import PIResponse
from core.classes.simulation.response import SimulationResponse
from core.classes.simulation.request_body import SimulationRequestBody
from middleware.simulate import simulate
from middleware.optimal_control import optimal_control
from middleware.parameters_identification import parameters_identification
from middleware.validate_expression import validate_expression

app: Flask = Flask(__name__)


@app.route('/simulate', methods=['POST'])
def simulate_endpoint() -> Response:
    """Simulation endpoint"""

    body: SimulationRequestBody = SimulationRequestBody(request.get_json())

    result: SimulationResponse = simulate(body.parameters, body.model)

    return jsonify(result.definition)


@app.route('/optimal-control', methods=['POST'])
def optimal_control_endpoint() -> Response:
    """Optimal Control endpoint"""

    body: OptimalControlRequestBody = OptimalControlRequestBody(
        request.get_json()
    )

    result: OptimalControlResponse = optimal_control(
        body.parameters, body.model
    )

    return jsonify(result.definition)


@app.route('/parameters-identification', methods=['POST'])
def parameters_identification_endpoint():
    """Parameters identification endpoint"""

    body: PIRequestBody = PIRequestBody(request.get_json())

    result: PIResponse = parameters_identification(body.parameters, body.model)

    return jsonify(result.definition)


@app.route('/validate-expression', methods=['POST'])
def validate_expression_endpoint() -> Response:
    """Expression validation endpoint"""

    body: ValidationRequestBody = ValidationRequestBody(request.get_json())

    result: ValidationResponse = validate_expression(
        body.expression, body.allowed_symbols
    )

    return jsonify(result.definition)


if __name__ == '__main__':
    app.run(debug=True)
