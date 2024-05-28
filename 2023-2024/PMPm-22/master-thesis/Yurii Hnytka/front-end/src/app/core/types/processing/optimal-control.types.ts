import {
    IErrorResponse,
    IRequestBody,
    ISuccessResponse,
    IValues,
} from './common.types';
import { ISimulationResult } from './simulation.types';

export interface InterventionParameters {
    nodesAmount: number;
    approximationType: ApproximationType;
    lowerBoundary: number;
    upperBoundary: number;
}

export interface IOptimalControlParameters {
    time: number;
    nodesAmount: number;
    objectiveFunction: string;
    intervention: InterventionParameters;
}

export interface IOptimalControlResult {
    compartments: IValues[];
    interventions: IValues[];
    approximatedInterventions: IValues[];
    noControlObjective: number;
    optimalObjective: number;
}

export type OptimalControlRequestBody = IRequestBody<IOptimalControlParameters>;

export type OptimalControlSuccessResponse = ISuccessResponse<
    IOptimalControlParameters,
    [ISimulationResult, IOptimalControlResult]
>;
export type OptimalControlResponse =
    | OptimalControlSuccessResponse
    | IErrorResponse;

export enum ApproximationType {
    PiecewiseConstant = 'piecewise-constant',
    PiecewiseLinear = 'piecewise-linear',
}
