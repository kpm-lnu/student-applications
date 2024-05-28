import {
    isErrorResponse,
    isRequestBody,
    isSuccessResponse,
    isValues,
} from './common.guards';
import {
    ISimulationParameters,
    ISimulationResult,
    SimulationRequestBody,
    SimulationResponse,
    SimulationSuccessResponse,
} from './simulation.types';

export function isSimulationParameters(
    simulationParameters: any
): simulationParameters is ISimulationParameters {
    const isKeysAmountValid: boolean =
        Object.keys(simulationParameters).length === 2;

    const isTimeValid: boolean =
        'time' in simulationParameters &&
        typeof simulationParameters.time === 'number';
    const isNodesAmountValid: boolean =
        'nodesAmount' in simulationParameters &&
        typeof simulationParameters.nodesAmount === 'number';

    return isKeysAmountValid && isTimeValid && isNodesAmountValid;
}

export function isSimulationResult(
    simulationResult: any
): simulationResult is ISimulationResult {
    const isKeysAmountValid: boolean =
        Object.keys(simulationResult).length === 1;

    const isCompartmentsValid: boolean =
        'compartments' in simulationResult &&
        Array.isArray(simulationResult.compartments) &&
        simulationResult.compartments.every((compartment: any): boolean =>
            isValues(compartment)
        );

    return isKeysAmountValid && isCompartmentsValid;
}

export function isSimulationRequestBody(
    simulationRequestBody: any
): simulationRequestBody is SimulationRequestBody {
    const isRequestBodyValid: boolean = isRequestBody(simulationRequestBody);

    const isParametersValid: boolean = isSimulationParameters(
        simulationRequestBody.parameters
    );

    return isRequestBodyValid && isParametersValid;
}

export function isSimulationSuccessResponse(
    simulationSuccessResponse: any
): simulationSuccessResponse is SimulationSuccessResponse {
    const isSuccessResponseValid: boolean = isSuccessResponse(
        simulationSuccessResponse
    );

    const isParametersValid: boolean = isSimulationParameters(
        simulationSuccessResponse.parameters
    );
    const isResultValid: boolean = isSimulationResult(
        simulationSuccessResponse.result
    );

    return isSuccessResponseValid && isParametersValid && isResultValid;
}

export function isSimulationResponse(
    simulationResponse: any
): simulationResponse is SimulationResponse {
    return (
        isSimulationSuccessResponse(simulationResponse) ||
        isErrorResponse(simulationResponse)
    );
}
