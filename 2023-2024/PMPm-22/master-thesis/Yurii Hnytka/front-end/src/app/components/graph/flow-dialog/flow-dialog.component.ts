import { Component, Inject } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { ICompartment, IFlow } from '@core/types/model.types';
import { IOption } from '@core/types/utils.types';
import { ModelService } from 'src/app/services/model.service';
import { v4 as uuid } from 'uuid';

@Component({
    selector: 'app-flow-dialog',
    templateUrl: './flow-dialog.component.html',
    styleUrls: ['./flow-dialog.component.scss'],
})
export class FlowDialogComponent {
    public readonly edit: boolean = false;

    public readonly control: FormGroup = new FormGroup({
        equation: new FormControl<string>('', [Validators.required]),
        source: new FormControl<string>('', [Validators.required]),
        target: new FormControl<string>('', [Validators.required]),
    });

    public get sources(): IOption[] {
        return this.modelService.compartments
            .filter(
                (compartment: ICompartment): boolean =>
                    compartment.id !== this.control.value.target
            )
            .map(
                (compartment: ICompartment): IOption => ({
                    value: compartment.id,
                    label: compartment.name,
                })
            );
    }

    public get targets(): IOption[] {
        return this.modelService.compartments
            .filter(
                (compartment: ICompartment): boolean =>
                    compartment.id !== this.control.value.source
            )
            .map(
                (compartment: ICompartment): IOption => ({
                    value: compartment.id,
                    label: compartment.name,
                })
            );
    }

    constructor(
        private readonly dialogRef: MatDialogRef<FlowDialogComponent, IFlow>,
        private readonly modelService: ModelService,
        @Inject(MAT_DIALOG_DATA) private readonly data?: IFlow
    ) {
        if (data) {
            this.edit = true;
            this.control.patchValue(data);
        }
    }

    public onClose(): void {
        this.dialogRef.close();
    }

    public onAccept(): void {
        this.dialogRef.close({
            id: this.edit ? this.data!.id : uuid(),
            ...this.control.value,
        });
    }

    public optionsTrackBy(_: number, option: IOption): string {
        return option.value;
    }
}
