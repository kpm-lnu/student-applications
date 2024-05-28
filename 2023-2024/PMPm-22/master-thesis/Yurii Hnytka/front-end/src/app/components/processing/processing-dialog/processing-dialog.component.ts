import { Component } from '@angular/core';
import { FormControl, FormGroup } from '@angular/forms';
import { MatDialogRef } from '@angular/material/dialog';
import {
    IOptimalControlParameters,
    IPIParameters,
    ISimulationParameters,
    Parameters,
    ProcessingType,
} from '@core/types/processing';

@Component({
    selector: 'app-processing-dialog',
    templateUrl: './processing-dialog.component.html',
    styleUrls: ['./processing-dialog.component.scss'],
})
export class ProcessingDialogComponent {
    public type: ProcessingType = ProcessingType.Simulation;

    public readonly control: FormGroup = new FormGroup({
        simulation: new FormControl<ISimulationParameters | null>(null),
        optimalControl: new FormControl<IOptimalControlParameters | null>(null),
        parametersIdentification: new FormControl<IPIParameters | null>(null),
    });

    public get currentControl(): FormControl {
        switch (this.type) {
            case ProcessingType.Simulation:
                return this.control.get('simulation') as FormControl;

            case ProcessingType.OptimalControl:
                return this.control.get('optimalControl') as FormControl;

            case ProcessingType.ParametersIdentification:
                return this.control.get(
                    'parametersIdentification'
                ) as FormControl;
        }
    }

    constructor(
        private readonly dialogRef: MatDialogRef<
            ProcessingDialogComponent,
            Parameters
        >
    ) {}

    public onClose(): void {
        this.dialogRef.close();
    }

    public onProcess(): void {
        this.dialogRef.close(this.currentControl.value);
    }
}
