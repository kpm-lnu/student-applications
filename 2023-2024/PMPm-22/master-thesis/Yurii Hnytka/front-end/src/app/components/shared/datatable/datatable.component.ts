import { Component, HostBinding, Input, ViewChild } from '@angular/core';
import {
    ControlValueAccessor,
    FormControl,
    FormGroup,
    NG_VALIDATORS,
    NG_VALUE_ACCESSOR,
    ValidationErrors,
    Validator,
    ValidatorFn,
} from '@angular/forms';
import { MatTable } from '@angular/material/table';
import {
    ColumnScheme,
    IRow,
    ISelectColumnScheme,
    InputType,
    RowScheme,
} from '@core/types/datatable.types';
import { IOption, OnChangeFn, OnTouchedFn } from '@core/types/utils.types';

@Component({
    selector: 'app-datatable',
    templateUrl: './datatable.component.html',
    styleUrls: ['./datatable.component.scss'],
    providers: [
        {
            provide: NG_VALUE_ACCESSOR,
            useExisting: DatatableComponent,
            multi: true,
        },
        {
            provide: NG_VALIDATORS,
            useExisting: DatatableComponent,
            multi: true,
        },
    ],
})
export class DatatableComponent<
    RowDataType extends Record<string, any>,
    DataType extends RowDataType
> implements ControlValueAccessor, Validator
{
    private _disabled: boolean = false;
    private _rowScheme: RowScheme<RowDataType> = {} as RowScheme<RowDataType>;
    private _rowValidators: ValidatorFn[] = [];

    private onChange!: OnChangeFn<void>;
    private onTouched!: OnTouchedFn<void>;

    private readonly optionsMap: Map<string, IOption[]> = new Map();

    private get data(): (DataType | RowDataType)[] {
        return this.rows.map(
            (row: IRow<DataType | RowDataType>): DataType | RowDataType =>
                structuredClone(row.data)
        );
    }

    @Input() public enumerate: boolean = false;
    @HostBinding('class.compact') @Input() public compact: boolean = false;
    @Input() public set disabled(disabled: boolean) {
        this.setDisabledState(disabled);
    }
    @Input() public set rowScheme(scheme: RowScheme<RowDataType>) {
        this._rowScheme = scheme;

        this.updateMaxRows();
        this.updateOptionsMap();
        this.initControl();
        this.initControlValidators();
    }
    public get rowScheme(): RowScheme<RowDataType> {
        return this._rowScheme;
    }
    @Input() public set rowValidators(validators: ValidatorFn[]) {
        this._rowValidators = validators;

        this.initControlValidators();
    }
    @Input('data') public set dataInput(data: DataType[] | null) {
        this.writeValue(data);
    }

    @ViewChild(MatTable) public table!: MatTable<IRow<DataType | RowDataType>>;

    public rows: IRow<DataType | RowDataType>[] = [];

    public adding: boolean = false;
    public editing: boolean = false;
    public maxRows: number = Infinity;

    public control: FormGroup = new FormGroup({});

    public get ids(): string[] {
        return Object.keys(this._rowScheme);
    }
    public get columns(): string[] {
        const columns: string[] = [];

        if (this.enumerate) {
            columns.push('index');
        }

        columns.push(...this.ids);

        if (!this.compact) {
            columns.push('placeholder');
        }

        if (!this._disabled) {
            columns.push('actions');
        }

        return columns;
    }

    public get editable(): boolean {
        return this.ids.some(
            (id: string): boolean => !!this._rowScheme[id].editable
        );
    }

    public writeValue(value: DataType[] | null): void {
        this.setData(value ?? []);

        this.updateOptionsMap();
    }

    public registerOnChange(
        onChange: OnChangeFn<DataType | RowDataType[]>
    ): void {
        this.onChange = (): void => {
            onChange(this.data);
        };
    }

    public registerOnTouched(onTouched: OnTouchedFn<void>): void {
        this.onTouched = onTouched;
    }

    public setDisabledState(disabled: boolean): void {
        this._disabled = disabled;
    }

    public validate(): ValidationErrors | null {
        if (!this.editing) {
            return null;
        }

        return {
            unsavedChanges: true,
        };
    }

    public getOptionLabel(id: string, value: string): string {
        const scheme: ISelectColumnScheme = this._rowScheme[
            id
        ] as ISelectColumnScheme;

        const option: IOption = scheme.options.find(
            (option: IOption): boolean => option.value === value
        ) as IOption;

        return option.label;
    }

    public getAvailableOptions(id: string): IOption[] {
        return this.optionsMap.get(id) || [];
    }

    public getErrors(id: string): string[] {
        const control: FormControl = this.control.get(id) as FormControl;

        const errors: ValidationErrors | null = control.errors;

        return errors ? Object.keys(errors) : [];
    }

    public onEdit(row: IRow<DataType | RowDataType>): void {
        this.control.patchValue(row.data);

        this.editing = true;
        row.editing = true;

        this.onTouched();
        this.onChange();
    }

    public onDelete(index: number): void {
        this.rows.splice(index, 1);

        this.updateOptionsMap();

        this.onTouched();
        this.onChange();
        this.table.renderRows();
    }

    public onSave(row: IRow<DataType | RowDataType>): void {
        this.ids.forEach((id: string): void => {
            const scheme: ColumnScheme = this._rowScheme[id];

            if (!scheme.editable && !this.adding) {
                return;
            }

            let value: any = this.control.get(id)?.value;

            switch (scheme.type) {
                case InputType.Number:
                    value = parseFloat(value);

                    break;
            }

            // @ts-ignore
            row.data[id] = value;
        });

        this.control.reset();
        this.updateOptionsMap();

        this.editing = false;
        row.editing = false;

        if (this.adding) {
            this.adding = false;

            this.disableNonEditableColumns();
        }

        this.onTouched();
        this.onChange();
    }

    public onCancel(row: IRow<DataType | RowDataType>): void {
        this.control.reset();

        this.editing = false;
        row.editing = false;

        if (this.adding) {
            this.adding = false;

            this.disableNonEditableColumns();

            this.rows.pop();
            this.table.renderRows();
        }

        this.onTouched();
        this.onChange();
    }

    public onRowAdd(): void {
        this.editing = true;
        this.adding = true;

        this.onChange();

        const newRow: IRow<RowDataType> = this.getEmptyRow();

        newRow.editing = true;

        this.disableNonEditableColumns(false);

        this.rows.push(newRow);
        this.table.renderRows();
    }

    private updateMaxRows(): void {
        const optionsCounts: number[] = this.ids.reduce(
            (counts: number[], id: string): number[] => {
                const scheme: ColumnScheme = this._rowScheme[id];

                if (scheme.type !== InputType.Select || !scheme.exclusive) {
                    return counts;
                }

                counts.push(scheme.options.length);

                return counts;
            },
            []
        );

        this.maxRows =
            optionsCounts.length > 0 ? Math.min(...optionsCounts) : Infinity;
    }

    private updateOptionsMap(): void {
        this.ids.forEach((id: string): void => {
            const scheme: ColumnScheme = this._rowScheme[id];

            if (scheme.type !== InputType.Select) {
                return;
            }

            if (!scheme.exclusive) {
                this.optionsMap.set(id, scheme.options);

                return;
            }

            const available: IOption[] = scheme.options.filter(
                (option: IOption): boolean =>
                    !this.rows.some(
                        (row: IRow<DataType | RowDataType>): boolean =>
                            option.value === row.data[id]
                    )
            );

            this.optionsMap.set(id, available);
        });
    }

    private initControl(): void {
        this.control = new FormGroup({});

        this.ids.forEach((id: string): void => {
            const scheme: ColumnScheme = this._rowScheme[id];

            const control: FormControl = new FormControl(
                null,
                scheme.validationFns || []
            );

            if (!scheme.editable) {
                control.disable();
            }

            this.control.addControl(id, control);
        });
    }

    private initControlValidators(): void {
        this.control.clearValidators();
        this.control.setValidators(this._rowValidators);
        this.control.updateValueAndValidity();
    }

    private setData(data: DataType[]): void {
        this.rows = data.map(
            (data: DataType): IRow<DataType> => ({
                data,
                editing: false,
            })
        );
    }

    private disableNonEditableColumns(disabled: boolean = true): void {
        this.ids.forEach((id: string): void => {
            const scheme: ColumnScheme = this._rowScheme[id];

            if (!scheme.editable) {
                const control: FormControl = this.control.get(
                    id
                ) as FormControl;

                if (disabled) {
                    control.disable();

                    return;
                }

                control.enable();
            }
        });
    }

    private getEmptyRow(): IRow<RowDataType> {
        return {
            data: this.ids.reduce(
                (data: RowDataType, id: string): RowDataType => ({
                    ...data,
                    [id]: null,
                }),
                {} as RowDataType
            ),
            editing: false,
        };
    }
}
