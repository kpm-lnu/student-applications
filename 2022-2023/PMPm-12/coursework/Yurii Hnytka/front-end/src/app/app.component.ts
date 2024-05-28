import { Component, ElementRef, ViewChild } from '@angular/core';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';
import cytoscape, {
    EdgeCollection,
    EdgeSingular,
    EventObject,
    NodeSingular,
    SingularElementArgument,
} from 'cytoscape';
import edgehandles from 'cytoscape-edgehandles';
import klay from 'cytoscape-klay';
import { Observable, Subscription, filter, map, tap } from 'rxjs';
import { CompartmentCreationDialogComponent } from './components/compartment-creation-dialog/compartment-creation-dialog.component';
import { ConfirmationDialogComponent } from './components/confirmation-dialog/confirmation-dialog.component';
import { DefinitionsTableDialogComponent } from './components/definitions-table-dialog/definitions-table-dialog.component';
import { FlowCreationDialogComponent } from './components/flow-creation-dialog/flow-creation-dialog.component';
import { SimulationDashboardComponent } from './components/simulation-dashboard/simulation-dashboard.component';
import { SimulationDialogComponent } from './components/simulation-dialog/simulation-dialog.component';
import { cytoscapeLayoutOptions, cytoscapeOptions } from './core/constants';
import {
    Definition,
    DefinitionType,
    ICompartment,
    ICompartmentBase,
    IConstant,
    IImportFlow,
    IImportModel,
    IIntervention,
    ISimulationParameters,
    ISimulationResults,
} from './core/interfaces';
import { DefinitionsService } from './services/definitions.service';
import { SamplesService } from './services/model-samples.service';
import { SimulationService } from './services/simulation.service';

interface ISimulationDialogOutput {
    simulationParameters: ISimulationParameters;
    costFunction: string;
    isOptimalControlProblem: boolean;
}

cytoscape.use(klay);
cytoscape.use(edgehandles);

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.scss'],
})
export class AppComponent {
    public simulationResults!: ISimulationResults;

    public get elementSelected(): boolean {
        return (
            this.cytoscapeObj &&
            this.cytoscapeObj.elements(':selected').length > 0
        );
    }

    public get compartmentsCount(): number {
        return this.cytoscapeObj && this.cytoscapeObj.nodes().length;
    }

    public get isModelValid(): boolean {
        return true;
    }

    @ViewChild('canvas') private readonly canvas!: ElementRef<HTMLElement>;

    private cytoscapeObj!: cytoscape.Core;
    private edgehandles!: edgehandles.EdgeHandlesInstance;

    private readonly subscriptions: Subscription = new Subscription();

    constructor(
        private readonly dialogService: MatDialog,
        private readonly simulationService: SimulationService,
        private readonly definitionsService: DefinitionsService,
        private readonly samplesService: SamplesService
    ) {}

    public ngAfterViewInit(): void {
        this.cytoscapeObj = cytoscape({
            container: this.canvas.nativeElement,
            ...cytoscapeOptions,
        });

        this.edgehandles = this.cytoscapeObj.edgehandles();

        this.cytoscapeObj.on('dblclick', (event: EventObject) => {
            if (!event.target.group) {
                return;
            }

            if (event.target.group() === 'nodes') {
                this.openEditCompartmentDialog();
            }

            if (event.target.group() === 'edges') {
                this.openEditFlowDialog();
            }
        });

        this.cytoscapeObj.on('cxttapend', 'node', (event: EventObject) => {
            this.edgehandles.start(event.target);
        });

        this.cytoscapeObj.on(
            'ehcomplete',
            this.onFlowCreationDialog.bind(this) as any
        );

        this.loadSample();
    }

    public openDefinitionsTableDialog(): void {
        const definitionsTableDialog: MatDialogRef<DefinitionsTableDialogComponent> =
            this.dialogService.open(DefinitionsTableDialogComponent, {
                autoFocus: false,
                maxWidth: '95vw',
                maxHeight: '95vh',
                height: '95vh',
            });
    }

    public openAddCompartmentDialog(): void {
        const creationDialog: MatDialogRef<CompartmentCreationDialogComponent> =
            this.dialogService.open(CompartmentCreationDialogComponent, {
                autoFocus: false,
            });

        this.subscriptions.add(
            creationDialog
                .afterClosed()
                .pipe(
                    filter((compartment: ICompartmentBase) =>
                        Boolean(compartment)
                    ),
                    tap(this.addCompartment.bind(this))
                )
                .subscribe()
        );
    }

    public layout(): void {
        this.cytoscapeObj.elements().layout(cytoscapeLayoutOptions).run();
    }

    public onDeleteElementDialog(): void {
        const selectedElement: SingularElementArgument = this.cytoscapeObj
            .elements(':selected')
            .first();

        const deletionSubject: string =
            selectedElement.group() === 'nodes' ? 'compartment' : 'flow';

        const deletionDialog: MatDialogRef<ConfirmationDialogComponent> =
            this.dialogService.open(ConfirmationDialogComponent, {
                data: {
                    title: `Delete ${deletionSubject}`,
                    message: `Are you sure you want to delete the selected ${deletionSubject}?`,
                },
                autoFocus: false,
            });

        this.subscriptions.add(
            deletionDialog
                .afterClosed()
                .pipe(
                    filter((confirmed: boolean) => confirmed),
                    tap(this.deleteElement.bind(this))
                )
                .subscribe()
        );
    }

    public onClearCompartmentsDialog(): void {
        const clearingDialog: MatDialogRef<ConfirmationDialogComponent> =
            this.dialogService.open(ConfirmationDialogComponent, {
                data: {
                    title: 'Clear model',
                    message:
                        'Are you sure you want to delete whole model?',
                },
                autoFocus: false,
            });

        this.subscriptions.add(
            clearingDialog
                .afterClosed()
                .pipe(
                    filter((confirmed: boolean) => confirmed),
                    tap(this.clearCompartments.bind(this))
                )
                .subscribe()
        );
    }

    public onSimulationResultsDialog(): void {
        const simulationResultsDialog: MatDialogRef<SimulationDashboardComponent> =
            this.dialogService.open(SimulationDashboardComponent, {
                autoFocus: false,
                data: this.simulationResults,
                height: '95vh',
                width: '95vw',
            });
    }

    public onModelSimulateDialog(): void {
        const simulationDialog: MatDialogRef<SimulationDialogComponent> =
            this.dialogService.open(SimulationDialogComponent, {
                autoFocus: false,
            });

        this.subscriptions.add(
            simulationDialog
                .afterClosed()
                .pipe(
                    filter((output: ISimulationDialogOutput) =>
                        Boolean(output)
                    ),
                    tap(this.simulateModel.bind(this))
                )
                .subscribe()
        );
    }

    private openEditCompartmentDialog(): void {
        const selectedCompartment: NodeSingular = this.cytoscapeObj
            .nodes(':selected')
            .first();

        const editDialog: MatDialogRef<CompartmentCreationDialogComponent> =
            this.dialogService.open(CompartmentCreationDialogComponent, {
                autoFocus: false,
                data: selectedCompartment.data(),
            });

        this.subscriptions.add(
            editDialog
                .afterClosed()
                .pipe(
                    filter((compartment: ICompartmentBase) =>
                        Boolean(compartment)
                    ),
                    tap(this.editCompartment.bind(this))
                )
                .subscribe()
        );
    }

    private onFlowCreationDialog(
        event: EventObject,
        sourceNode: NodeSingular,
        targetNode: NodeSingular,
        addedEdge: EdgeSingular
    ): void {
        const creationDialog: MatDialogRef<FlowCreationDialogComponent> =
            this.dialogService.open(FlowCreationDialogComponent, {
                autoFocus: false,
            });

        this.subscriptions.add(
            creationDialog
                .afterClosed()
                .pipe(
                    tap((flowEquation: string) => {
                        if (!flowEquation) {
                            addedEdge.remove();
                        }
                    }),
                    filter((flowEquation: string) => Boolean(flowEquation)),
                    map((flowEquation: string): [string, EdgeSingular] => [
                        flowEquation,
                        addedEdge,
                    ]),
                    tap(this.addFlow.bind(this))
                )
                .subscribe()
        );
    }

    private openEditFlowDialog(): void {
        const selectedFlow: EdgeSingular = this.cytoscapeObj
            .edges(':selected')
            .first();

        const editDialog: MatDialogRef<FlowCreationDialogComponent> =
            this.dialogService.open(FlowCreationDialogComponent, {
                autoFocus: false,
                data: selectedFlow.data('equation'),
            });

        this.subscriptions.add(
            editDialog
                .afterClosed()
                .pipe(
                    filter((flowEquation: string) => Boolean(flowEquation)),
                    tap(this.editFlow.bind(this))
                )
                .subscribe()
        );
    }

    private addCompartment(compartmentData: ICompartmentBase): void {
        this.cytoscapeObj.add({
            group: 'nodes',
            data: {
                id: compartmentData.id,
                value: compartmentData.value,
            },
            renderedPosition: {
                x: this.cytoscapeObj.width() / 2,
                y: this.cytoscapeObj.height() / 2,
            },
        });

        this.definitionsService.addDefinition({
            id: compartmentData.id,
            type: DefinitionType.Compartment,
            value: compartmentData.value,
        });
    }

    private editCompartment(compartmentData: ICompartmentBase): void {
        const selectedCompartment: NodeSingular = this.cytoscapeObj
            .nodes(':selected')
            .first();

        selectedCompartment.data('id', compartmentData.id);
        selectedCompartment.data('value', compartmentData.value);
    }

    private addFlow([flowEquation, flowEdge]: [string, EdgeSingular]): void {
        flowEdge.data('equation', flowEquation);
    }

    private editFlow(flowEquation: string): void {
        const selectedFlow: EdgeSingular = this.cytoscapeObj
            .edges(':selected')
            .first();

        selectedFlow.data('equation', flowEquation);
    }

    private deleteElement(): void {
        this.cytoscapeObj.remove(':selected');
    }

    private clearCompartments(): void {
        this.cytoscapeObj.remove('node');
    }

    private simulateModel(simulationData: ISimulationDialogOutput): void {
        let observable: Observable<ISimulationResults>;

        if (!simulationData.isOptimalControlProblem) {
            observable = this.simulationService.simulateModel({
                model: this.getCompartments(),
                simulationParameters: {
                    step: simulationData.simulationParameters.step,
                    time: simulationData.simulationParameters.time,
                },
            });
        } else {
            const constants: IConstant[] = this.getConstants();

            observable = this.simulationService.optimizeModel({
                model: this.getCompartments(),
                simulationParameters: {
                    step: simulationData.simulationParameters.step,
                    time: simulationData.simulationParameters.time,
                },
                costFunction: constants.reduce(
                    (costFunction: string, constant: IConstant) => {
                        return costFunction.replace(
                            new RegExp(`\\b${constant.id}\\b`),
                            constant.value.toString()
                        );
                    },
                    simulationData.costFunction
                ),
                interventions: this.getInterventions(),
            });
        }

        observable
            .pipe(
                tap((results: ISimulationResults) => {
                    this.simulationResults = results;
                })
            )
            .subscribe();
    }

    private getCompartments(): ICompartment[] {
        const constants: IConstant[] = this.getConstants();

        return this.cytoscapeObj.nodes().map(
            (node: NodeSingular): ICompartment => ({
                id: node.id(),
                value: node.data('value'),
                inflows: this.getFlows(node.incomers().edges()).map(
                    (flow: string) => {
                        constants.forEach((constant: IConstant) => {
                            flow = flow.replace(
                                new RegExp(`\\b${constant.id}\\b`),
                                constant.value.toString()
                            );
                        });

                        return flow;
                    }
                ),
                outflows: this.getFlows(node.outgoers().edges()).map(
                    (flow: string) => {
                        constants.forEach((constant: IConstant) => {
                            flow = flow.replace(
                                new RegExp(`\\b${constant.id}\\b`),
                                constant.value.toString()
                            );
                        });

                        return flow;
                    }
                ),
            })
        );
    }

    private getConstants(): IConstant[] {
        return this.definitionsService.getDefinitionsTable().constants.map(
            (constant: IConstant): IConstant => ({
                id: constant.id,
                value: constant.value,
            })
        );
    }

    private getInterventions(): IIntervention[] {
        return this.definitionsService.getDefinitionsTable().interventions.map(
            (intervention: IIntervention): IIntervention => ({
                id: intervention.id,
            })
        );
    }

    private getFlows(flowEdges: EdgeCollection): string[] {
        return flowEdges.map((flowEdge: EdgeSingular): string =>
            flowEdge.data('equation')
        );
    }

    private loadSample(): void {
        this.samplesService
            .getSample('default')
            .pipe(
                tap(this.parseSample.bind(this)),
                tap(() => {
                    this.layout();
                })
            )
            .subscribe();
    }

    private parseSample(sample: IImportModel): void {
        this.clearCompartments();

        sample.compartments.forEach((compartment: ICompartmentBase) => {
            this.addCompartment(compartment);
        });

        sample.flows.forEach((flow: IImportFlow) => {
            this.cytoscapeObj.add({
                group: 'edges',
                data: {
                    source: flow.source,
                    target: flow.target,
                    equation: flow.equation,
                },
            });
        });

        sample.definitions.forEach((definition: Definition) => {
            this.definitionsService.addDefinition(definition);
        });
    }
}
