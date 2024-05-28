import { Component, OnDestroy } from '@angular/core';
import {
    AbstractControl,
    ControlValueAccessor,
    FormControl,
    FormGroup,
    NG_VALIDATORS,
    NG_VALUE_ACCESSOR,
    ValidationErrors,
    Validator,
    ValidatorFn,
    Validators,
} from '@angular/forms';
import { MatSnackBar } from '@angular/material/snack-bar';
import { InputType, RowScheme } from '@core/types/datatable.types';
import { SelectedConstantDefinition } from '@core/types/definitions.types';
import {
    ICompartment,
    IConstant,
    IIntervention,
} from '@core/types/model.types';
import {
    IPIParameters,
    ISelectedConstant,
    IValues,
} from '@core/types/processing';
import { IOption, OnChangeFn, OnTouchedFn } from '@core/types/utils.types';
import { rowDataToValues, valuesToRowData } from '@core/utils';
import { Subscription, tap } from 'rxjs';
import { FilesService } from 'src/app/services/files.service';
import { ModelService } from 'src/app/services/model.service';

@Component({
    selector: 'app-parameters-identification-parameters-input-panel',
    templateUrl:
        './parameters-identification-parameters-input-panel.component.html',
    styleUrls: [
        './parameters-identification-parameters-input-panel.component.scss',
    ],
    providers: [
        {
            provide: NG_VALUE_ACCESSOR,
            useExisting: ParametersIdentificationParametersInputPanelComponent,
            multi: true,
        },
        {
            provide: NG_VALIDATORS,
            useExisting: ParametersIdentificationParametersInputPanelComponent,
            multi: true,
        },
    ],
})
export class ParametersIdentificationParametersInputPanelComponent
    implements ControlValueAccessor, Validator, OnDestroy
{
    private readonly subscription: Subscription = new Subscription();

    public readonly control: FormGroup = new FormGroup({
        timeStep: new FormControl<number | null>(null, [Validators.required]),
        selectedConstants: new FormControl<ISelectedConstant[] | null>(null, [
            Validators.required,
        ]),
        data: new FormControl<IValues[] | null>(null, [Validators.required]),
    });

    public readonly selectedConstantsControl: FormControl<
        SelectedConstantDefinition[]
    > = new FormControl<SelectedConstantDefinition[]>([]) as FormControl<
        SelectedConstantDefinition[]
    >;
    public readonly dataControl: FormControl<Record<string, number>[]> =
        new FormControl<Record<string, number>[]>([]) as FormControl<
            Record<string, number>[]
        >;

    public readonly constantsRowScheme: RowScheme<SelectedConstantDefinition> =
        {
            name: {
                name: 'Name',
                type: InputType.Select,
                exclusive: true,
                options: this.modelService.constants.map(
                    (constant: IConstant): IOption => ({
                        value: constant.name,
                        label: constant.name,
                    })
                ),
                validationFns: [Validators.required],
            },
            lowerBoundary: {
                name: 'Lower boundary',
                type: InputType.Number,
                validationFns: [Validators.required],
                editable: true,
            },
            value: {
                name: 'Initial guess',
                type: InputType.Number,
                validationFns: [Validators.required],
                editable: true,
            },
            upperBoundary: {
                name: 'Upper boundary',
                type: InputType.Number,
                validationFns: [Validators.required],
                editable: true,
            },
        };

    public readonly dataRowScheme: RowScheme = [
        ...this.modelService.compartments.map(
            (compartment: ICompartment): RowScheme => ({
                [compartment.name]: {
                    name: compartment.name,
                    editable: true,
                    type: InputType.Number,
                    validationFns: [Validators.min(0)],
                },
            })
        ),
        ...this.modelService.interventions.map(
            (intervention: IIntervention): RowScheme => ({
                [intervention.name]: {
                    name: intervention.name,
                    editable: true,
                    type: InputType.Number,
                    validationFns: [Validators.required],
                },
            })
        ),
    ].reduce(
        (rowScheme: RowScheme, columnScheme: RowScheme): RowScheme => ({
            ...rowScheme,
            ...columnScheme,
        }),
        {}
    );
    public readonly dataRowValidators: ValidatorFn[] = [
        (control: AbstractControl): ValidationErrors | null => {
            const value: Record<string, number> = structuredClone(
                control.value
            );

            return Object.values(value).some((value: number): boolean => {
                if (!value) {
                    return false;
                }

                return true;
            })
                ? null
                : { missingValues: true };
        },
    ];

    public constructor(
        private readonly snackBar: MatSnackBar,
        private readonly filesService: FilesService,
        private readonly modelService: ModelService
    ) {
        this.subscription.add(
            this.selectedConstantsControl.valueChanges
                .pipe(tap(this.onSelectedConstantsChange.bind(this)))
                .subscribe()
        );

        this.subscription.add(
            this.dataControl.valueChanges
                .pipe(tap(this.onDataChange.bind(this)))
                .subscribe()
        );
    }

    public ngOnDestroy(): void {
        this.subscription.unsubscribe();
    }

    public writeValue(value: IPIParameters | null): void {
        this.control.setValue(
            value ?? {
                timeStep: null,
                selectedConstants: null,
                data: null,
            }
        );

        this.selectedConstantsControl.setValue([]);
        this.dataControl.setValue([]);
    }

    public registerOnChange(onChange: OnChangeFn<IPIParameters>): void {
        this.subscription.add(this.control.valueChanges.subscribe(onChange));
    }

    public registerOnTouched(onTouched: OnTouchedFn<IPIParameters>): void {
        this.subscription.add(this.control.valueChanges.subscribe(onTouched));
    }

    public setDisabledState(disabled: boolean): void {
        if (disabled) {
            this.control.disable();
            this.selectedConstantsControl.disable();
            this.dataControl.disable();

            return;
        }

        this.control.enable();
        this.selectedConstantsControl.enable();
        this.dataControl.enable();
    }

    public validate(): ValidationErrors | null {
        return this.control.valid ? null : { invalid: true };
    }

    public onDataImport(): void {
        this.subscription.add(
            this.filesService
                .readDataFromFile<IValues[]>('.scs')
                .pipe(
                    tap((data: IValues[]): void => {
                        this.dataControl.setValue(valuesToRowData(data));

                        this.snackBar.open('Imported successfully', 'Dismiss', {
                            panelClass: 'snackbar',
                            horizontalPosition: 'right',
                            verticalPosition: 'bottom',
                            duration: 5000,
                        });
                    })
                )
                .subscribe()
        );
    }

    private onSelectedConstantsChange(
        selectedConstants: SelectedConstantDefinition[]
    ): void {
        const constants: IConstant[] = this.modelService.constants;

        this.control.controls['selectedConstants']!.setValue(
            selectedConstants.map(
                (
                    selectedConstant: SelectedConstantDefinition
                ): ISelectedConstant =>
                    ({
                        ...selectedConstant,
                        id: constants.find(
                            (constant: IConstant): boolean =>
                                constant.name === selectedConstant.name
                        )!.id,
                    } as ISelectedConstant)
            )
        );
    }

    private onDataChange(rowData: Record<string, number>[]): void {
        this.control.controls['data']!.setValue(rowDataToValues(rowData));
    }
}
