import { Component, Inject } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { ICompartmentBase } from 'src/app/core/interfaces';

@Component({
    selector: 'app-compartment-creation-dialog',
    templateUrl: './compartment-creation-dialog.component.html',
    styleUrls: ['./compartment-creation-dialog.component.scss'],
})
export class CompartmentCreationDialogComponent {
    public readonly formGroup: FormGroup = new FormGroup({
        id: new FormControl(null, [Validators.required]),
        value: new FormControl(null, [Validators.required]),
    });

    constructor(
        private readonly dialogRef: MatDialogRef<CompartmentCreationDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data?: ICompartmentBase
    ) {
        if (data) {
            this.formGroup.patchValue(data);
        }
    }

    public closeDialog(): void {
        this.dialogRef.close(null);
    }

    public addCompartment(): void {
        this.dialogRef.close(this.formGroup.value as ICompartmentBase);
    }
}
