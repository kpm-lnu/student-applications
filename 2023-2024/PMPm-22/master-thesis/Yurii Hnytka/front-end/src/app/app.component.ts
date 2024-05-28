import {
    AfterViewInit,
    Component,
    ElementRef,
    OnDestroy,
    OnInit,
    ViewChild,
} from '@angular/core';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';
import { MatSnackBar } from '@angular/material/snack-bar';
import { ModelDefinition } from '@core/classes/model.class';

import { DataDefinition } from '@core/types/definitions.types';
import {
    ICompartment,
    IConstant,
    IFlow,
    IIntervention,
    IModel,
} from '@core/types/model.types';
import {
    ISelectedConstant,
    OptimalControlResponse,
    PIResponse,
    Parameters,
    SimulationResponse,
} from '@core/types/processing';
import { isSuccessResponse } from '@core/types/processing/common.guards';
import { isOptimalControlParameters } from '@core/types/processing/optimal-control.guards';
import { isPIParameters } from '@core/types/processing/parameters-identification.guards';
import { isSimulationParameters } from '@core/types/processing/simulation.guards';
import { EdgeSingular, NodeSingular, SingularElementArgument } from 'cytoscape';
import {
    Observable,
    Subscription,
    concat,
    debounceTime,
    filter,
    tap,
} from 'rxjs';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { DefinitionsTableDialogComponent } from './components/definitions-table-dialog/definitions-table-dialog.component';
import { CompartmentDialogComponent } from './components/graph/compartment-dialog/compartment-dialog.component';
import { FlowDialogComponent } from './components/graph/flow-dialog/flow-dialog.component';
import { ProcessingDialogComponent } from './components/processing/processing-dialog/processing-dialog.component';
import { ConfirmationDialogComponent } from './components/shared/confirmation-dialog/confirmation-dialog.component';
import { fromResizeObserver } from './core/utils';
import { FilesService } from './services/files.service';
import { ModelService } from './services/model.service';
import { ProcessingService } from './services/processing.service';
import { RunsService } from './services/runs.service';
import { WorkspacesService } from './services/workspaces.service';

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.scss'],
})
export class AppComponent implements OnInit, AfterViewInit, OnDestroy {
    @ViewChild('canvas') private readonly canvas!: ElementRef<HTMLDivElement>;

    private readonly windowResize$: Observable<ResizeObserverEntry> =
        fromResizeObserver(window.document.body);
    private readonly subscriptions: Subscription = new Subscription();

    public get selected(): boolean {
        return (
            this.modelService.graphReady &&
            !!this.modelService.selectedElement.length
        );
    }
    public get compartmentsCount(): number {
        return this.modelService.compartments.length;
    }

    constructor(
        private readonly dialogService: MatDialog,
        private readonly snackBar: MatSnackBar,
        private readonly filesService: FilesService,
        public readonly workspacesService: WorkspacesService,
        private readonly modelService: ModelService,
        private readonly processingService: ProcessingService,
        public readonly runsService: RunsService
    ) {}

    public ngOnInit(): void {
        const windowResizeSub: Subscription = this.windowResize$
            .pipe(debounceTime(500), tap(this.onLayout.bind(this)))
            .subscribe();

        this.subscriptions.add(windowResizeSub);
    }

    public ngAfterViewInit(): void {
        this.modelService.initGraph(this.canvas.nativeElement);
        this.initWorkspaces();

        const compartmentOpenSub: Subscription =
            this.modelService.compartmentOpen
                .pipe(tap(this.onCompartmentEdit.bind(this)))
                .subscribe();

        const flowAddSub: Subscription = this.modelService.flowAdd
            .pipe(tap(this.onFlowAdd.bind(this)))
            .subscribe();

        const flowOpenSub: Subscription = this.modelService.flowOpen
            .pipe(tap(this.onFlowEdit.bind(this)))
            .subscribe();

        this.subscriptions.add(compartmentOpenSub);
        this.subscriptions.add(flowAddSub);
        this.subscriptions.add(flowOpenSub);
    }

    public ngOnDestroy(): void {
        this.subscriptions.unsubscribe();
    }

    public onDefinitionsTable(): void {
        const dialog: MatDialogRef<
            DefinitionsTableDialogComponent,
            ModelDefinition
        > = this.dialogService.open(DefinitionsTableDialogComponent, {
            autoFocus: false,
            disableClose: true,
        });

        this.subscriptions.add(
            dialog
                .afterClosed()
                .pipe(
                    filter(
                        (
                            definition?: ModelDefinition
                        ): definition is ModelDefinition => !!definition
                    ),
                    tap(this.modelService.update.bind(this.modelService))
                )
                .subscribe()
        );
    }

    public onLayout(): void {
        this.modelService.layoutGraph();
    }

    public onClear(): void {
        const dialog: MatDialogRef<ConfirmationDialogComponent, boolean> =
            this.dialogService.open(ConfirmationDialogComponent, {
                autoFocus: false,
                disableClose: true,
                data: {
                    title: 'Clear model',
                    message:
                        'Are you sure you want to delete whole model?' +
                        '\n\nAll constants will remain intact',
                },
            });

        this.subscriptions.add(
            dialog
                .afterClosed()
                .pipe(
                    filter((confirmed?: boolean): boolean => !!confirmed),
                    tap((): void => {
                        this.modelService.clear();
                    })
                )
                .subscribe()
        );
    }

    public onImport(): void {
        this.subscriptions.add(
            this.filesService
                .readDataFromFile<IModel>('.scm')
                .pipe(
                    tap((definition: IModel): void => {
                        this.workspacesService.replaceModel(definition);

                        this.snackBar.open(
                            'Model imported successfully',
                            'Dismiss',
                            {
                                panelClass: 'snackbar',
                                horizontalPosition: 'right',
                                verticalPosition: 'bottom',
                                duration: 5000,
                            }
                        );
                    })
                )
                .subscribe()
        );
    }

    public onExport(): void {
        this.filesService.downloadFileWithData(
            this.modelService.model,
            `${this.workspacesService.current.name}.scm`
        );

        this.snackBar.open('Model export is ready for downloading', 'Dismiss', {
            panelClass: 'snackbar',
            horizontalPosition: 'right',
            verticalPosition: 'bottom',
            duration: 5000,
        });
    }

    public onDashboard(): void {
        this.dialogService.open(DashboardComponent, {
            autoFocus: false,
            disableClose: true,
            maxWidth: '100vw',
            maxHeight: '100vh',
        });
    }

    public onProcess(): void {
        const dialog: MatDialogRef<ProcessingDialogComponent, Parameters> =
            this.dialogService.open(ProcessingDialogComponent, {
                autoFocus: false,
                disableClose: true,
            });

        this.subscriptions.add(
            dialog
                .afterClosed()
                .pipe(
                    filter(
                        (parameters?: Parameters): parameters is Parameters =>
                            !!parameters
                    ),
                    tap(this.processModel.bind(this))
                )
                .subscribe()
        );
    }

    public onDelete(): void {
        const selectedElement: SingularElementArgument =
            this.modelService.selectedElement;

        const isCompartment: boolean = selectedElement.group() === 'nodes';
        const id: string = selectedElement.id();

        const deletionSubject: string = isCompartment ? 'compartment' : 'flow';

        const dialog: MatDialogRef<ConfirmationDialogComponent, boolean> =
            this.dialogService.open(ConfirmationDialogComponent, {
                autoFocus: false,
                disableClose: true,
                data: {
                    title: `Delete ${deletionSubject}`,
                    message: `Are you sure you want to delete the selected ${deletionSubject}?`,
                },
            });

        this.subscriptions.add(
            dialog
                .afterClosed()
                .pipe(
                    filter((confirmed?: boolean): boolean => !!confirmed),
                    tap((): void => {
                        if (isCompartment) {
                            this.modelService.removeCompartment(id);

                            return;
                        }

                        this.modelService.removeFlow(id);
                    })
                )
                .subscribe()
        );
    }

    public onCompartmentAdd(): void {
        const dialog: MatDialogRef<CompartmentDialogComponent, ICompartment> =
            this.dialogService.open(CompartmentDialogComponent, {
                autoFocus: false,
                disableClose: true,
            });

        this.subscriptions.add(
            dialog
                .afterClosed()
                .pipe(
                    filter(
                        (
                            definition?: ICompartment
                        ): definition is ICompartment => !!definition
                    ),
                    tap(
                        this.modelService.addCompartment.bind(this.modelService)
                    )
                )
                .subscribe()
        );
    }

    private onCompartmentEdit(node: NodeSingular): void {
        const dialog: MatDialogRef<CompartmentDialogComponent, ICompartment> =
            this.dialogService.open(CompartmentDialogComponent, {
                autoFocus: false,
                disableClose: true,
                data: node.data(),
            });

        this.subscriptions.add(
            dialog
                .afterClosed()
                .pipe(
                    filter(
                        (
                            definition?: ICompartment
                        ): definition is ICompartment => !!definition
                    ),
                    tap(
                        this.modelService.updateCompartment.bind(
                            this.modelService
                        )
                    )
                )
                .subscribe()
        );
    }

    private onFlowAdd(edge: EdgeSingular): void {
        const dialog: MatDialogRef<FlowDialogComponent, IFlow> =
            this.dialogService.open(FlowDialogComponent, {
                autoFocus: false,
                data: {
                    source: edge.source().id(),
                    target: edge.target().id(),
                },
            });

        this.subscriptions.add(
            dialog
                .afterClosed()
                .pipe(
                    tap((definition?: IFlow): void => {
                        if (!definition) {
                            edge.remove();
                        }
                    }),
                    filter(
                        (definition?: IFlow): definition is IFlow =>
                            !!definition
                    ),
                    tap(this.modelService.updateFlow.bind(this.modelService))
                )
                .subscribe()
        );
    }

    private onFlowEdit(edge: EdgeSingular): void {
        const dialog: MatDialogRef<FlowDialogComponent, IFlow> =
            this.dialogService.open(FlowDialogComponent, {
                autoFocus: false,
                data: edge.data(),
            });

        this.subscriptions.add(
            dialog
                .afterClosed()
                .pipe(
                    filter(
                        (definition?: IFlow): definition is IFlow =>
                            !!definition
                    ),
                    tap(this.modelService.updateFlow.bind(this.modelService))
                )
                .subscribe()
        );
    }

    private processModel(parameters: Parameters): void {
        let observable: Observable<
            SimulationResponse | OptimalControlResponse | PIResponse
        >;

        if (isSimulationParameters(parameters)) {
            observable = this.processingService.simulateModel({
                parameters,
                model: {
                    compartments: this.modelService.compartments,
                    constants: [
                        ...this.modelService.constants,
                        ...this.modelService.interventions.map(
                            (intervention: IIntervention): IConstant => ({
                                id: intervention.id,
                                name: intervention.name,
                                value: 0,
                            })
                        ),
                    ],
                    interventions: [],
                    flows: this.modelService.flows,
                },
            });
        } else if (isOptimalControlParameters(parameters)) {
            observable = this.processingService.optimizeModel({
                parameters,
                model: this.modelService.model,
            });
        } else if (isPIParameters(parameters)) {
            observable = this.processingService.identifyParameters({
                parameters,
                model: {
                    compartments: this.modelService.compartments,
                    constants: this.modelService.constants.filter(
                        (constant: IConstant): boolean =>
                            !parameters.selectedConstants.find(
                                (
                                    selectedConstant: ISelectedConstant
                                ): boolean =>
                                    selectedConstant.id === constant.id
                            )
                    ),
                    interventions: [
                        ...this.modelService.interventions,
                        ...parameters.selectedConstants.map(
                            (
                                selectedConstant: ISelectedConstant
                            ): IIntervention => ({
                                id: selectedConstant.id,
                                name: selectedConstant.name,
                            })
                        ),
                    ],
                    flows: this.modelService.flows,
                },
            });
        } else {
            return;
        }

        this.subscriptions.add(
            observable
                .pipe(
                    tap(
                        (
                            response:
                                | SimulationResponse
                                | OptimalControlResponse
                                | PIResponse
                        ): void => {
                            if (isSuccessResponse(response)) {
                                this.runsService.add({
                                    parameters: response.parameters,
                                    result: response.result,
                                } as DataDefinition);

                                this.snackBar.open(
                                    'Processing completed successfully',
                                    'Dismiss',
                                    {
                                        panelClass: 'snackbar',
                                        horizontalPosition: 'right',
                                        verticalPosition: 'bottom',
                                        duration: 5000,
                                    }
                                );

                                return;
                            }

                            this.snackBar.open(
                                'Processing failed.\n\n' +
                                    `Error: ${response.error}`,
                                'Dismiss',
                                {
                                    panelClass: 'snackbar',
                                    horizontalPosition: 'right',
                                    verticalPosition: 'bottom',
                                }
                            );
                        }
                    )
                )
                .subscribe()
        );
    }

    private initWorkspaces(): void {
        this.subscriptions.add(
            concat(
                this.workspacesService.addSample('sir.scm'),
                this.workspacesService.addSample('interventions.scm')
            ).subscribe()
        );
    }
}
