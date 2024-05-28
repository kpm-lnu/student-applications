import {
    OptimalControlSuccessResponse,
    PISuccessResponse,
    SimulationSuccessResponse,
} from './processing';

export interface IRun<DataType = any> {
    name: string;
    data: DataType;
}

export type SimulationData = SimulationSuccessResponse & {
    type: DataType.Simulation;
};
export type OptimalControlData = OptimalControlSuccessResponse & {
    type: DataType.OptimalControl;
};
export type PIData = PISuccessResponse & {
    type: DataType.PI;
};
export type NoneData = {
    type: DataType.None;
};
export type Data = SimulationData | OptimalControlData | PIData | NoneData;

export type SimulationRun = IRun<SimulationData>;
export type OptimalControlRun = IRun<OptimalControlData>;
export type PIRun = IRun<PIData>;
export type NoneRun = IRun<NoneData>;
export type Run = SimulationRun | OptimalControlRun | PIRun | NoneRun;

export enum DataType {
    Simulation = 'Simulation',
    OptimalControl = 'OptimalControl',
    PI = 'PI',
    None = 'None',
}
