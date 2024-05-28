import { Component, Inject } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { ICompartment } from '@core/types/model.types';
import { ValidationService } from 'src/app/services/validation.service';
import { v4 as uuid } from 'uuid';

@Component({
    selector: 'app-compartment-dialog',
    templateUrl: './compartment-dialog.component.html',
    styleUrls: ['./compartment-dialog.component.scss'],
})
export class CompartmentDialogComponent {
    public readonly edit: boolean = false;

    public readonly control: FormGroup = new FormGroup({
        name: new FormControl<string>('', [
            Validators.required,
            this.validationService.getDefinitionNameValidator(this.data?.name),
        ]),
        value: new FormControl<number>(0, [
            Validators.required,
            Validators.min(0),
        ]),
    });

    constructor(
        private readonly dialogRef: MatDialogRef<
            CompartmentDialogComponent,
            ICompartment
        >,
        private readonly validationService: ValidationService,
        @Inject(MAT_DIALOG_DATA) public data?: ICompartment
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
}
