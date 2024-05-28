import {
    isErrorResponse,
    isRequestBody,
    isSuccessResponse,
    isValues,
} from './common.guards';
import {
    IIdentifiedConstant,
    IPIParameters,
    IPIResult,
    ISelectedConstant,
    PIRequestBody,
    PIResponse,
    PISuccessResponse,
} from './parameters-identification.types';

export function isSelectedConstant(
    selectedConstant: any
): selectedConstant is ISelectedConstant {
    const isKeysAmountValid: boolean =
        Object.keys(selectedConstant).length === 5;

    const isIdValid: boolean =
        'id' in selectedConstant && typeof selectedConstant.id === 'string';
    const isNameValid: boolean =
        'name' in selectedConstant && typeof selectedConstant.name === 'string';
    const isUpperBoundaryValid: boolean =
        'upperBoundary' in selectedConstant &&
        typeof selectedConstant.upperBoundary === 'number';
    const isValueValid: boolean =
        'value' in selectedConstant &&
        typeof selectedConstant.value === 'number';
    const isLowerBoundaryValid: boolean =
        'lowerBoundary' in selectedConstant &&
        typeof selectedConstant.lowerBoundary === 'number';

    return (
        isKeysAmountValid &&
        isIdValid &&
        isNameValid &&
        isUpperBoundaryValid &&
        isValueValid &&
        isLowerBoundaryValid
    );
}

export function isIdentifiedConstant(
    identifiedConstant: any
): identifiedConstant is IIdentifiedConstant {
    const isKeysAmountValid: boolean =
        Object.keys(identifiedConstant).length === 3;

    const isIdValid: boolean =
        'id' in identifiedConstant && typeof identifiedConstant.id === 'string';
    const isNameValid: boolean =
        'name' in identifiedConstant &&
        typeof identifiedConstant.name === 'string';
    const isValueValid: boolean =
        'value' in identifiedConstant &&
        typeof identifiedConstant.value === 'number';

    return isKeysAmountValid && isIdValid && isNameValid && isValueValid;
}

export function isPIParameters(
    piParameters: any
): piParameters is IPIParameters {
    const isKeysAmountValid: boolean = Object.keys(piParameters).length === 3;

    const isTimeStepValid: boolean =
        'timeStep' in piParameters && typeof piParameters.timeStep === 'number';
    const isSelectedConstantsValid: boolean =
        'selectedConstants' in piParameters &&
        Array.isArray(piParameters.selectedConstants) &&
        piParameters.selectedConstants.every((selectedConstant: any): boolean =>
            isSelectedConstant(selectedConstant)
        );
    const isDataValid: boolean =
        'data' in piParameters &&
        Array.isArray(piParameters.data) &&
        piParameters.data.every((value: any): boolean => isValues(value));

    return (
        isKeysAmountValid &&
        isSelectedConstantsValid &&
        isTimeStepValid &&
        isDataValid
    );
}

export function isPIResult(piResult: any): piResult is IPIResult {
    const isKeysAmountValid: boolean = Object.keys(piResult).length === 2;

    const isConstantsValid: boolean =
        'constants' in piResult &&
        Array.isArray(piResult.constants) &&
        piResult.constants.every((constant: any): boolean =>
            isIdentifiedConstant(constant)
        );
    const isApproximationValid: boolean =
        'approximation' in piResult &&
        Array.isArray(piResult.approximation) &&
        piResult.approximation.every((value: any): boolean => isValues(value));

    return isKeysAmountValid && isConstantsValid && isApproximationValid;
}

export function isPIRequestBody(
    piRequestBody: any
): piRequestBody is PIRequestBody {
    const isRequestBodyValid: boolean = isRequestBody(piRequestBody);

    const isParametersValid: boolean = isPIParameters(piRequestBody.parameters);

    return isRequestBodyValid && isParametersValid;
}

export function isPISuccessResponse(
    piSuccessResponse: any
): piSuccessResponse is PISuccessResponse {
    const isSuccessResponseValid: boolean =
        isSuccessResponse(piSuccessResponse);

    const isParametersValid: boolean = isPIParameters(
        piSuccessResponse.parameters
    );
    const isResultValid: boolean = isPIResult(piSuccessResponse.result);

    return isSuccessResponseValid && isParametersValid && isResultValid;
}

export function isPIResponse(piResponse: any): piResponse is PIResponse {
    return isPISuccessResponse(piResponse) || isErrorResponse(piResponse);
}
