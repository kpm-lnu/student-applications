import { Component, Inject } from '@angular/core';
import { FormControl, ValidationErrors, Validators } from '@angular/forms';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';

@Component({
    selector: 'app-flow-creation-dialog',
    templateUrl: './flow-creation-dialog.component.html',
    styleUrls: ['./flow-creation-dialog.component.scss'],
})
export class FlowCreationDialogComponent {
    public readonly flowEquationFormControl: FormControl = new FormControl(
        null,
        [Validators.required]
    );

    constructor(
        private readonly dialogRef: MatDialogRef<FlowCreationDialogComponent>,
        @Inject(MAT_DIALOG_DATA) public data?: string
    ) {
        if (data) {
            this.flowEquationFormControl.setValue(data);
        }
    }

    public onErrorsChange(validationErrors: ValidationErrors | null): void {
        this.flowEquationFormControl.setErrors(validationErrors);
    }

    public closeDialog(): void {
        this.dialogRef.close(null);
    }

    public addCompartment(): void {
        this.dialogRef.close(this.flowEquationFormControl.value);
    }
}
