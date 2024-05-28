import {
    ICompartment,
    IConstant,
    IFlow,
    IIntervention,
    IModel,
} from '@core/types/model.types';

export class ModelDefinition {
    private _compartments: ICompartment[] = [];
    private _constants: IConstant[] = [];
    private _interventions: IIntervention[] = [];
    private _flows: IFlow[] = [];

    public get compartments(): ICompartment[] {
        return structuredClone(this._compartments);
    }
    public get constants(): IConstant[] {
        return structuredClone(this._constants);
    }
    public get interventions(): IIntervention[] {
        return structuredClone(this._interventions);
    }
    public get flows(): IFlow[] {
        return structuredClone(this._flows);
    }
    public get model(): IModel {
        return {
            compartments: this.compartments,
            constants: this.constants,
            interventions: this.interventions,
            flows: this.flows,
        };
    }

    public get symbols(): string[] {
        return [
            ...this._compartments.map(
                (compartment: ICompartment): string => compartment.name
            ),
            ...this._constants.map(
                (constant: IConstant): string => constant.name
            ),
            ...this._interventions.map(
                (intervention: IIntervention): string => intervention.name
            ),
        ];
    }

    public constructor(config?: IModel) {
        if (config) {
            this._compartments = config.compartments;
            this._constants = config.constants;
            this._interventions = config.interventions;
            this._flows = config.flows;
        }
    }

    public clear(constants: boolean = false): void {
        this._compartments = [];
        this._interventions = [];
        this._flows = [];

        if (constants) {
            this._constants = [];
        }
    }

    public addCompartment(compartment: ICompartment): void {
        this._compartments.push(compartment);
    }

    public updateCompartment(newCompartment: ICompartment): void {
        const index: number = this._compartments.findIndex(
            (compartment: ICompartment): boolean =>
                compartment.id === newCompartment.id
        );

        const compartment: ICompartment = this._compartments[index];

        this.renameDefinitionInFlows(compartment.name, newCompartment.name);

        this._compartments[index] = newCompartment;
    }

    public removeCompartment(id: string): void {
        const index: number = this._compartments.findIndex(
            (compartment: ICompartment): boolean => compartment.id === id
        );

        this._compartments.splice(index, 1);
    }

    public addConstant(constant: IConstant): void {
        this._constants.push(constant);
    }

    public updateConstant(newConstant: IConstant): void {
        const index: number = this._constants.findIndex(
            (constant: IConstant): boolean => constant.id === newConstant.id
        );

        const constant: IConstant = this._constants[index];

        this.renameDefinitionInFlows(constant.name, newConstant.name);

        this._constants[index] = newConstant;
    }

    public removeConstant(id: string): void {
        const index: number = this._constants.findIndex(
            (constant: IConstant): boolean => constant.id === id
        );

        this._constants.splice(index, 1);
    }

    public addIntervention(intervention: IIntervention): void {
        this._interventions.push(intervention);
    }

    public updateIntervention(newIntervention: IIntervention): void {
        const index: number = this._interventions.findIndex(
            (intervention: IIntervention): boolean =>
                intervention.id === newIntervention.id
        );

        const intervention: IIntervention = this._interventions[index];

        this.renameDefinitionInFlows(intervention.name, newIntervention.name);

        this._interventions[index] = newIntervention;
    }

    public removeIntervention(id: string): void {
        const index: number = this._interventions.findIndex(
            (intervention: IIntervention): boolean => intervention.id === id
        );

        this._interventions.splice(index, 1);
    }

    public addFlow(flow: IFlow): void {
        this._flows.push(flow);
    }

    public updateFlow(newFlow: IFlow): void {
        const index: number = this._flows.findIndex(
            (flow: IFlow): boolean => flow.id === newFlow.id
        );

        this._flows[index] = newFlow;
    }

    public removeFlow(id: string): void {
        const index: number = this._flows.findIndex(
            (flow: IFlow): boolean => flow.id === id
        );

        this._flows.splice(index, 1);
    }

    private renameDefinitionInFlows(name: string, newName: string): void {
        const expressionRegExp: RegExp = new RegExp(`\\b${name}\\b`, 'g');

        this._flows.forEach((flow: IFlow): void => {
            flow.equation = flow.equation.replace(expressionRegExp, newName);
        });
    }
}
