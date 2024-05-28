import {
    isErrorResponse,
    isRequestBody,
    isSuccessResponse,
    isValues,
} from './common.guards';
import {
    ApproximationType,
    IOptimalControlParameters,
    IOptimalControlResult,
    InterventionParameters,
    OptimalControlSuccessResponse,
} from './optimal-control.types';
import { isSimulationResult } from './simulation.guards';

export function isInterventionParameters(
    interventionParameters: any
): interventionParameters is InterventionParameters {
    const isKeysAmountValid: boolean =
        Object.keys(interventionParameters).length === 4;

    const isNodesAmountValid: boolean =
        'nodesAmount' in interventionParameters &&
        typeof interventionParameters.nodesAmount === 'number';
    const isApproximationTypeValid: boolean =
        'approximationType' in interventionParameters &&
        isApproximationType(interventionParameters.approximationType);
    const isLowerBoundaryValid: boolean =
        'lowerBoundary' in interventionParameters &&
        typeof interventionParameters.lowerBoundary === 'number';
    const isUpperBoundaryValid: boolean =
        'upperBoundary' in interventionParameters &&
        typeof interventionParameters.upperBoundary === 'number';

    return (
        isKeysAmountValid &&
        isNodesAmountValid &&
        isApproximationTypeValid &&
        isLowerBoundaryValid &&
        isUpperBoundaryValid
    );
}

export function isOptimalControlParameters(
    optimalControlParameters: any
): optimalControlParameters is IOptimalControlParameters {
    const isKeysAmountValid: boolean =
        Object.keys(optimalControlParameters).length === 4;

    const isTimeValid: boolean =
        'time' in optimalControlParameters &&
        typeof optimalControlParameters.time === 'number';
    const isNodesAmountValid: boolean =
        'nodesAmount' in optimalControlParameters &&
        typeof optimalControlParameters.nodesAmount === 'number';
    const isObjectiveFunctionValid: boolean =
        'objectiveFunction' in optimalControlParameters &&
        typeof optimalControlParameters.objectiveFunction === 'string';
    const isInterventionParametersValid: boolean =
        'intervention' in optimalControlParameters &&
        isInterventionParameters(optimalControlParameters.intervention);

    return (
        isKeysAmountValid &&
        isTimeValid &&
        isNodesAmountValid &&
        isObjectiveFunctionValid &&
        isInterventionParametersValid
    );
}

export function isOptimalControlResult(
    optimalControlResult: any
): optimalControlResult is IOptimalControlResult {
    const isKeysAmountValid: boolean =
        Object.keys(optimalControlResult).length === 5;

    const isCompartmentsValid: boolean =
        'compartments' in optimalControlResult &&
        Array.isArray(optimalControlResult.compartments) &&
        optimalControlResult.compartments.every((compartment: any): boolean =>
            isValues(compartment)
        );
    const isInterventionsValid: boolean =
        'interventions' in optimalControlResult &&
        Array.isArray(optimalControlResult.interventions) &&
        optimalControlResult.interventions.every((intervention: any): boolean =>
            isValues(intervention)
        );
    const isApproximatedInterventionsValid: boolean =
        'approximatedInterventions' in optimalControlResult &&
        Array.isArray(optimalControlResult.approximatedInterventions) &&
        optimalControlResult.approximatedInterventions.every(
            (intervention: any): boolean => isValues(intervention)
        );
    const isNoControlObjectiveValid: boolean =
        'noControlObjective' in optimalControlResult &&
        typeof optimalControlResult.noControlObjective === 'number';
    const isOptimalObjectiveValid: boolean =
        'optimalObjective' in optimalControlResult &&
        typeof optimalControlResult.optimalObjective === 'number';

    return (
        isKeysAmountValid &&
        isCompartmentsValid &&
        isInterventionsValid &&
        isApproximatedInterventionsValid &&
        isNoControlObjectiveValid &&
        isOptimalObjectiveValid
    );
}

export function isOptimalControlRequestBody(
    optimalControlRequestBody: any
): optimalControlRequestBody is IOptimalControlParameters {
    const isRequestDataValid: boolean = isRequestBody(
        optimalControlRequestBody
    );

    const isParametersValid: boolean = isOptimalControlParameters(
        optimalControlRequestBody.parameters
    );

    return isRequestDataValid && isParametersValid;
}

export function isOptimalControlSuccessResponse(
    optimalControlSuccessResponse: any
): optimalControlSuccessResponse is OptimalControlSuccessResponse {
    const isSuccessResponseValid: boolean = isSuccessResponse(
        optimalControlSuccessResponse
    );

    const isParametersValid: boolean = isOptimalControlParameters(
        optimalControlSuccessResponse.parameters
    );
    const isResultsValid: boolean =
        optimalControlSuccessResponse.result.length === 2 &&
        isSimulationResult(optimalControlSuccessResponse.result[0]) &&
        isOptimalControlResult(optimalControlSuccessResponse.result[1]);

    return isSuccessResponseValid && isParametersValid && isResultsValid;
}

export function isOptimalControlResponse(
    optimalControlResponse: any
): optimalControlResponse is IOptimalControlResult {
    return (
        isOptimalControlSuccessResponse(optimalControlResponse) ||
        isErrorResponse(optimalControlResponse)
    );
}

export function isApproximationType(
    approximationType: any
): approximationType is ApproximationType {
    return Object.values(ApproximationType).includes(approximationType);
}
