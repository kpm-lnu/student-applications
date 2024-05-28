import { Component, OnInit } from '@angular/core';
import {
    FormControl,
    FormGroup,
    ValidationErrors,
    Validators,
} from '@angular/forms';
import { MatDialogRef } from '@angular/material/dialog';
import { ModelDefinition } from '@core/classes/model.class';
import { InputType, RowScheme } from '@core/types/datatable.types';
import {
    CompartmentDefinition,
    ConstantDefinition,
    FlowDefinition,
    InterventionDefinition,
} from '@core/types/definitions.types';
import {
    isCompartment,
    isConstant,
    isFlow,
    isIntervention,
} from '@core/types/model.guards';
import {
    ICompartment,
    IConstant,
    IFlow,
    IIntervention,
    IModel,
} from '@core/types/model.types';
import { IOption } from '@core/types/utils.types';
import { ModelService } from 'src/app/services/model.service';
import { ValidationService } from 'src/app/services/validation.service';
import { v4 as uuid } from 'uuid';

@Component({
    selector: 'app-definitions-table-dialog',
    templateUrl: './definitions-table-dialog.component.html',
    styleUrls: ['./definitions-table-dialog.component.scss'],
})
export class DefinitionsTableDialogComponent implements OnInit {
    public readonly compartmentsRowScheme: RowScheme<CompartmentDefinition> = {
        name: {
            name: 'Name',
            type: InputType.Text,
            editable: true,
            validationFns: [
                Validators.required,
                (control: FormControl): ValidationErrors | null =>
                    this.validationService.definitionName(control.value),
            ],
        },
        value: {
            name: 'Value',
            type: InputType.Number,
            editable: true,
            validationFns: [Validators.required, Validators.min(0)],
        },
    };

    public readonly constantsRowScheme: RowScheme<ConstantDefinition> = {
        name: {
            name: 'Name',
            type: InputType.Text,
            editable: true,
            validationFns: [
                Validators.required,
                (control: FormControl): ValidationErrors | null =>
                    this.validationService.definitionName(control.value),
            ],
        },
        value: {
            name: 'Value',
            type: InputType.Number,
            editable: true,
            validationFns: [Validators.required],
        },
    };

    public readonly interventionsRowScheme: RowScheme<InterventionDefinition> =
        {
            name: {
                name: 'Name',
                type: InputType.Text,
                editable: true,
                validationFns: [
                    Validators.required,
                    (control: FormControl): ValidationErrors | null =>
                        this.validationService.definitionName(control.value),
                ],
            },
        };

    public readonly flowsRowScheme: RowScheme<FlowDefinition> = {
        source: {
            name: 'Source',
            type: InputType.Select,
            exclusive: false,
            options: this.modelService.compartments.map(
                (compartment: ICompartment): IOption => ({
                    value: compartment.id,
                    label: compartment.name,
                })
            ),
            editable: true,
        },
        target: {
            name: 'Target',
            type: InputType.Select,
            exclusive: false,
            options: this.modelService.compartments.map(
                (compartment: ICompartment): IOption => ({
                    value: compartment.id,
                    label: compartment.name,
                })
            ),
            editable: true,
        },
    };

    public readonly control: FormGroup = new FormGroup({
        compartments: new FormControl<(ICompartment | CompartmentDefinition)[]>(
            []
        ),
        constants: new FormControl<(IConstant | ConstantDefinition)[]>([]),
        interventions: new FormControl<
            (IIntervention | InterventionDefinition)[]
        >([]),
        flows: new FormControl<(IFlow | FlowDefinition)[]>([]),
    });

    constructor(
        private readonly dialogRef: MatDialogRef<
            DefinitionsTableDialogComponent,
            ModelDefinition
        >,
        private readonly modelService: ModelService,
        private readonly validationService: ValidationService
    ) {}

    public ngOnInit(): void {
        this.init(this.modelService.model);
    }

    public onClose(): void {
        this.dialogRef.close();
    }

    public onSave(): void {
        const value: {
            compartments: (ICompartment | CompartmentDefinition)[];
            constants: (IConstant | ConstantDefinition)[];
            interventions: (IIntervention | InterventionDefinition)[];
            flows: (IFlow | FlowDefinition)[];
        } = this.control.value;

        const definition: ModelDefinition = new ModelDefinition({
            compartments: value.compartments.map(
                (
                    compartment: ICompartment | CompartmentDefinition
                ): ICompartment => {
                    if (isCompartment(compartment)) {
                        return compartment;
                    }

                    return {
                        ...compartment,
                        id: uuid(),
                    };
                }
            ),
            constants: value.constants.map(
                (constant: IConstant | ConstantDefinition): IConstant => {
                    if (isConstant(constant)) {
                        return constant;
                    }

                    return {
                        ...constant,
                        id: uuid(),
                    };
                }
            ),
            interventions: value.interventions.map(
                (
                    intervention: IIntervention | InterventionDefinition
                ): IIntervention => {
                    if (isIntervention(intervention)) {
                        return intervention;
                    }

                    return {
                        ...intervention,
                        id: uuid(),
                    };
                }
            ),
            flows: value.flows.map((flow: IFlow | FlowDefinition): IFlow => {
                if (isFlow(flow)) {
                    return flow;
                }

                return {
                    ...flow,
                    id: uuid(),
                    equation: '',
                };
            }),
        });

        this.dialogRef.close(definition);
    }

    private init(model: IModel): void {
        this.control.setValue(model);
    }
}
