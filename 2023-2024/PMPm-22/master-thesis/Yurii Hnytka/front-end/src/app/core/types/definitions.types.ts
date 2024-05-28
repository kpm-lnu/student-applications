import { ICompartment, IConstant, IFlow, IIntervention } from './model.types';
import { IIdentifiedConstant, ISelectedConstant } from './processing';
import { OptimalControlData, PIData, SimulationData } from './run.types';

export type CompartmentDefinition = Omit<ICompartment, 'id'>;
export type ConstantDefinition = Omit<IConstant, 'id'>;
export type InterventionDefinition = Omit<IIntervention, 'id'>;
export type FlowDefinition = Omit<IFlow, 'id' | 'equation'>;

export type SelectedConstantDefinition = Omit<ISelectedConstant, 'id'>;
export type IdentifiedConstantDefinition = Omit<IIdentifiedConstant, 'id'>;

export type SimulationDataDefinition = Omit<SimulationData, 'type'>;
export type OptimalControlDataDefinition = Omit<OptimalControlData, 'type'>;
export type PIDataDefinition = Omit<PIData, 'type'>;

export type DataDefinition =
    | SimulationDataDefinition
    | OptimalControlDataDefinition
    | PIDataDefinition;
