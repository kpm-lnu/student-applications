export interface ICompartmentBase {
    id: string;
    value: number;
}

export interface ICompartment extends ICompartmentBase {
    inflows: string[];
    outflows: string[];
}

export interface ISimulationParameters {
    time: number;
    step: number;
}

export interface IConstant {
    id: string;
    value: number;
}

export interface IIntervention {
    id: string;
}

export interface ISimulationData {
    model: ICompartment[];
    simulationParameters: ISimulationParameters;
}

export interface IOptimalControlData extends ISimulationData {
    interventions: IIntervention[];
    costFunction: string;
}

export interface ICompartmentSimulatedData {
    id: string;
    values: number[];
}

export interface ISimulationResults {
    time: number;
    step: number;
    compartments: ICompartmentSimulatedData[];
}

export enum DefinitionType {
    Compartment = 'compartment',
    Constant = 'constant',
    Intervention = 'intervention',
}

export type ICompartmentDefinition = ICompartmentBase & {
    type: DefinitionType.Compartment;
};

export type IConstantDefinition = IConstant & {
    type: DefinitionType.Constant;
};

export type IInterventionDefinition = IIntervention & {
    type: DefinitionType.Intervention;
};

export type Definition =
    | ICompartmentDefinition
    | IConstantDefinition
    | IInterventionDefinition;

export interface IDefinitionsTable {
    compartments: ICompartmentDefinition[];
    constants: IConstantDefinition[];
    interventions: IInterventionDefinition[];
}

export interface IImportFlow {
    source: string;
    target: string;
    equation: string;
}

export interface IImportModel {
    compartments: ICompartmentBase[];
    flows: IImportFlow[];
    definitions: Definition[];
}

export interface IValidationResponse {
    isValid: boolean;
    message?: string;
}
