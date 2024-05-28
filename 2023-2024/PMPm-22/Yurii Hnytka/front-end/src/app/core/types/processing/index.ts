import { IOptimalControlParameters } from './optimal-control.types';
import { IPIParameters } from './parameters-identification.types';
import { ISimulationParameters } from './simulation.types';

export * from './common.types';
export * from './dialog.types';
export * from './optimal-control.types';
export * from './parameters-identification.types';
export * from './simulation.types';

export type Parameters =
    | ISimulationParameters
    | IOptimalControlParameters
    | IPIParameters;
