import {
    IErrorResponse,
    IRequestBody,
    ISuccessResponse,
    IValues,
} from './common.types';

export interface ISimulationParameters {
    time: number;
    nodesAmount: number;
}

export interface ISimulationResult {
    compartments: IValues[];
}

export type SimulationRequestBody = IRequestBody<ISimulationParameters>;

export type SimulationSuccessResponse = ISuccessResponse<
    ISimulationParameters,
    ISimulationResult
>;
export type SimulationResponse = SimulationSuccessResponse | IErrorResponse;
