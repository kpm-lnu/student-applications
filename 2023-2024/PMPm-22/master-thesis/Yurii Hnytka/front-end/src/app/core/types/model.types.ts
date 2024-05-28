export interface ICompartment {
    id: string;
    name: string;
    value: number;
}

export interface IConstant {
    id: string;
    name: string;
    value: number;
}

export interface IIntervention {
    id: string;
    name: string;
}

export interface IFlow {
    id: string;
    source: string;
    target: string;
    equation: string;
}

export interface IModel {
    compartments: ICompartment[];
    constants: IConstant[];
    interventions: IIntervention[];
    flows: IFlow[];
}
