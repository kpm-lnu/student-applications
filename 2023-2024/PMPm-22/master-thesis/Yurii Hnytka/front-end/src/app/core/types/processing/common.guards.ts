import { isModel } from '../model.guards';
import {
    IErrorResponse,
    IRequestBody,
    ISuccessResponse,
    IValidationResponse,
    IValues,
} from './common.types';

export function isValues(values: any): values is IValues {
    const isKeysAmountValid: boolean = Object.keys(values).length === 2;

    const isNameValid: boolean =
        'name' in values && typeof values.name === 'string';
    const isValuesValid: boolean =
        'values' in values &&
        Array.isArray(values.values) &&
        values.values.every((value: any): boolean => typeof value === 'number');

    return isKeysAmountValid && isNameValid && isValuesValid;
}

export function isRequestBody(requestBody: any): requestBody is IRequestBody {
    const isKeysAmountValid: boolean = Object.keys(requestBody).length === 2;

    const isParametersValid: boolean = 'parameters' in requestBody;
    const isModelValid: boolean =
        'model' in requestBody && isModel(requestBody.model);

    return isKeysAmountValid && isParametersValid && isModelValid;
}

export function isErrorResponse(
    errorResponse: any
): errorResponse is IErrorResponse {
    const isKeysAmountValid: boolean = Object.keys(errorResponse).length === 1;

    const isErrorValid: boolean =
        'error' in errorResponse && typeof errorResponse.error === 'string';

    return isKeysAmountValid && isErrorValid;
}

export function isSuccessResponse(
    successResponse: any
): successResponse is ISuccessResponse {
    const isKeysAmountValid: boolean =
        Object.keys(successResponse).length === 2;

    const isParametersValid: boolean = 'parameters' in successResponse;
    const isResultValid: boolean = 'result' in successResponse;

    return isKeysAmountValid && isParametersValid && isResultValid;
}

export function isValidationResponse(
    validationResponse: any
): validationResponse is IValidationResponse {
    const keysAmount: number = Object.keys(validationResponse).length;

    const isKeysAmountValid: boolean = keysAmount <= 2;

    const isValidValid: boolean =
        'isValid' in validationResponse &&
        typeof validationResponse.isValid === 'boolean';

    if (keysAmount === 2) {
        const isMessageValid: boolean =
            'message' in validationResponse &&
            typeof validationResponse.message === 'string';

        return isKeysAmountValid && isValidValid && isMessageValid;
    }

    return isKeysAmountValid && isValidValid;
}
