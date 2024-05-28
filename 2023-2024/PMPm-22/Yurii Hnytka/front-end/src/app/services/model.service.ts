import { EventEmitter, Injectable } from '@angular/core';
import { ModelDefinition } from '@core/classes/model.class';
import {
    ICompartment,
    IConstant,
    IFlow,
    IIntervention,
    IModel,
} from '@core/types/model.types';
import { EdgeSingular, NodeSingular, SingularElementArgument } from 'cytoscape';
import { GraphService } from './graph.service';

@Injectable({
    providedIn: 'root',
})
export class ModelService {
    private _model: ModelDefinition = new ModelDefinition();

    public get compartmentOpen(): EventEmitter<NodeSingular> {
        return this.graphService.compartmentOpen;
    }
    public get flowAdd(): EventEmitter<EdgeSingular> {
        return this.graphService.flowAdd;
    }
    public get flowOpen(): EventEmitter<EdgeSingular> {
        return this.graphService.flowOpen;
    }

    public get graphReady(): boolean {
        return this.graphService.graphReady;
    }
    public get selectedElement(): SingularElementArgument {
        return this.graphService.selectedElement;
    }

    public get compartments(): ICompartment[] {
        return this._model.compartments;
    }
    public get constants(): IConstant[] {
        return this._model.constants;
    }
    public get interventions(): IIntervention[] {
        return this._model.interventions;
    }
    public get flows(): IFlow[] {
        return this._model.flows;
    }
    public get model(): IModel {
        return this._model.model;
    }

    public get symbols(): string[] {
        return this._model.symbols;
    }

    public constructor(private readonly graphService: GraphService) {}

    public initGraph(container: HTMLElement): void {
        this.graphService.init(container);
    }

    public layoutGraph(): void {
        this.graphService.layout();
    }

    public set(model: ModelDefinition): void {
        this._model = model;

        this.graphService.clear();

        model.compartments.forEach((compartment: ICompartment): void =>
            this.graphService.addCompartment(compartment, false)
        );

        model.flows.forEach((flow: IFlow): void =>
            this.graphService.addFlow(flow, false)
        );

        this.graphService.layout();
    }

    public update(newModel: ModelDefinition): void {
        this.updateCompartments(newModel.compartments, false);
        this.updateConstants(newModel.constants);
        this.updateInterventions(newModel.interventions);
        this.updateFlows(newModel.flows, false);

        this.graphService.layout();
    }

    public clear(removeConstants: boolean = false): void {
        this._model.clear(removeConstants);
        this.graphService.clear();
    }

    public updateCompartments(
        newCompartments: ICompartment[],
        layout: boolean = true
    ): void {
        const compartments: ICompartment[] = this.compartments;

        const {
            addedCompartments,
            updatedCompartments,
        }: {
            addedCompartments: ICompartment[];
            updatedCompartments: ICompartment[];
        } = newCompartments.reduce(
            (
                accumulator: {
                    addedCompartments: ICompartment[];
                    updatedCompartments: ICompartment[];
                },
                newCompartment: ICompartment
            ): {
                addedCompartments: ICompartment[];
                updatedCompartments: ICompartment[];
            } => {
                if (
                    compartments.find(
                        (compartment: ICompartment): boolean =>
                            compartment.id === newCompartment.id
                    )
                ) {
                    accumulator.updatedCompartments.push(newCompartment);

                    return accumulator;
                }

                accumulator.addedCompartments.push(newCompartment);

                return accumulator;
            },
            { addedCompartments: [], updatedCompartments: [] }
        );
        const removedCompartments: ICompartment[] = compartments.filter(
            (compartment: ICompartment): boolean =>
                !newCompartments.find(
                    (newCompartment: ICompartment): boolean =>
                        newCompartment.id === compartment.id
                )
        );

        addedCompartments.forEach((compartment: ICompartment): void => {
            this.addCompartment(compartment, false);
        });

        updatedCompartments.forEach((compartment: ICompartment): void => {
            this.updateCompartment(compartment);
        });

        removedCompartments.forEach((compartment: ICompartment): void => {
            this.removeCompartment(compartment.id, false);
        });

        if (layout) {
            this.graphService.layout();
        }
    }

    public addCompartment(
        compartment: ICompartment,
        layout: boolean = true
    ): void {
        this._model.addCompartment(compartment);
        this.graphService.addCompartment(compartment, layout);
    }

    public updateCompartment(newCompartment: ICompartment): void {
        this._model.updateCompartment(newCompartment);
        this.graphService.updateCompartment(newCompartment);
    }

    public removeCompartment(id: string, layout: boolean = true): void {
        this._model.removeCompartment(id);
        this.graphService.removeCompartment(id, layout);
    }

    public updateConstants(newConstants: IConstant[]): void {
        const constants: IConstant[] = this.constants;

        const {
            addedConstants,
            updatedConstants,
        }: {
            addedConstants: IConstant[];
            updatedConstants: IConstant[];
        } = newConstants.reduce(
            (
                accumulator: {
                    addedConstants: IConstant[];
                    updatedConstants: IConstant[];
                },
                newConstant: IConstant
            ): {
                addedConstants: IConstant[];
                updatedConstants: IConstant[];
            } => {
                if (
                    constants.find(
                        (constant: IConstant): boolean =>
                            constant.id === newConstant.id
                    )
                ) {
                    accumulator.updatedConstants.push(newConstant);

                    return accumulator;
                }

                accumulator.addedConstants.push(newConstant);

                return accumulator;
            },
            { addedConstants: [], updatedConstants: [] }
        );
        const removedConstants: IConstant[] = constants.filter(
            (constant: IConstant): boolean =>
                !newConstants.find(
                    (newConstant: IConstant): boolean =>
                        newConstant.id === constant.id
                )
        );

        addedConstants.forEach((constant: IConstant): void => {
            this.addConstant(constant);
        });

        updatedConstants.forEach((constant: IConstant): void => {
            this.updateConstant(constant);
        });

        removedConstants.forEach((constant: IConstant): void => {
            this.removeConstant(constant.id);
        });
    }

    public addConstant(constant: IConstant): void {
        this._model.addConstant(constant);
    }

    public updateConstant(newConstant: IConstant): void {
        this._model.updateConstant(newConstant);
    }

    public removeConstant(id: string): void {
        this._model.removeConstant(id);
    }

    public updateInterventions(newInterventions: IIntervention[]): void {
        const interventions: IIntervention[] = this.interventions;

        const {
            addedInterventions,
            updatedInterventions,
        }: {
            addedInterventions: IIntervention[];
            updatedInterventions: IIntervention[];
        } = newInterventions.reduce(
            (
                accumulator: {
                    addedInterventions: IIntervention[];
                    updatedInterventions: IIntervention[];
                },
                newIntervention: IIntervention
            ): {
                addedInterventions: IIntervention[];
                updatedInterventions: IIntervention[];
            } => {
                if (
                    interventions.find(
                        (intervention: IIntervention): boolean =>
                            intervention.id === newIntervention.id
                    )
                ) {
                    accumulator.updatedInterventions.push(newIntervention);

                    return accumulator;
                }

                accumulator.addedInterventions.push(newIntervention);

                return accumulator;
            },
            { addedInterventions: [], updatedInterventions: [] }
        );
        const removedInterventions: IIntervention[] = interventions.filter(
            (intervention: IIntervention): boolean =>
                !newInterventions.find(
                    (newIntervention: IIntervention): boolean =>
                        newIntervention.id === intervention.id
                )
        );

        addedInterventions.forEach((intervention: IIntervention): void => {
            this.addIntervention(intervention);
        });

        updatedInterventions.forEach((intervention: IIntervention): void => {
            this.updateIntervention(intervention);
        });

        removedInterventions.forEach((intervention: IIntervention): void => {
            this.removeIntervention(intervention.id);
        });
    }

    public addIntervention(intervention: IIntervention): void {
        this._model.addIntervention(intervention);
    }

    public updateIntervention(newIntervention: IIntervention): void {
        this._model.updateIntervention(newIntervention);
    }

    public removeIntervention(id: string): void {
        this._model.removeIntervention(id);
    }

    public updateFlows(newFlows: IFlow[], layout: boolean = true): void {
        const flows: IFlow[] = this.flows;

        const {
            addedFlows,
            updatedFlows,
        }: {
            addedFlows: IFlow[];
            updatedFlows: IFlow[];
        } = newFlows.reduce(
            (
                accumulator: {
                    addedFlows: IFlow[];
                    updatedFlows: IFlow[];
                },
                newFlow: IFlow
            ): {
                addedFlows: IFlow[];
                updatedFlows: IFlow[];
            } => {
                if (
                    flows.find((flow: IFlow): boolean => flow.id === newFlow.id)
                ) {
                    accumulator.updatedFlows.push(newFlow);

                    return accumulator;
                }

                accumulator.addedFlows.push(newFlow);

                return accumulator;
            },
            { addedFlows: [], updatedFlows: [] }
        );
        const removedFlows: IFlow[] = flows.filter(
            (flow: IFlow): boolean =>
                !newFlows.find(
                    (newFlow: IFlow): boolean => newFlow.id === flow.id
                )
        );

        addedFlows.forEach((flow: IFlow): void => {
            this.addFlow(flow, false);
        });

        updatedFlows.forEach((flow: IFlow): void => {
            this.updateFlow(flow, false);
        });

        removedFlows.forEach((flow: IFlow): void => {
            this.removeFlow(flow.id, false);
        });

        if (layout) {
            this.graphService.layout();
        }
    }

    public addFlow(flow: IFlow, layout: boolean = true): void {
        this._model.addFlow(flow);
        this.graphService.addFlow(flow, layout);
    }

    public updateFlow(newFlow: IFlow, layout: boolean = true): void {
        this._model.updateFlow(newFlow);
        this.graphService.updateFlow(newFlow, layout);
    }

    public removeFlow(id: string, layout: boolean = true): void {
        this._model.removeFlow(id);
        this.graphService.removeFlow(id, layout);
    }
}
