import {
    IErrorResponse,
    IRequestBody,
    ISuccessResponse,
    IValues,
} from './common.types';

export interface ISelectedConstant {
    id: string;
    name: string;
    upperBoundary: number;
    value: number;
    lowerBoundary: number;
}

export interface IIdentifiedConstant {
    id: string;
    name: string;
    value: number;
}

export interface IPIParameters {
    timeStep: number;
    selectedConstants: ISelectedConstant[];
    data: IValues[];
}

export interface IPIResult {
    constants: IIdentifiedConstant[];
    approximation: IValues[];
}

export type PIRequestBody = IRequestBody<IPIParameters>;

export type PISuccessResponse = ISuccessResponse<IPIParameters, IPIResult>;
export type PIResponse = PISuccessResponse | IErrorResponse;
