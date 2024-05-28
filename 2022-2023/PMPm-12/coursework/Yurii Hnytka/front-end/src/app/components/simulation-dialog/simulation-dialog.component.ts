import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { MatDialogRef } from '@angular/material/dialog';
import { tap } from 'rxjs';

@Component({
    selector: 'app-simulation-dialog',
    templateUrl: './simulation-dialog.component.html',
    styleUrls: ['./simulation-dialog.component.scss'],
})
export class SimulationDialogComponent implements OnInit {
    public readonly formGroup: FormGroup = new FormGroup({
        simulationParameters: new FormGroup({
            time: new FormControl(null, [Validators.required]),
            step: new FormControl(null, [Validators.required]),
        }),
        costFunction: new FormControl(null, [Validators.required]),
        isOptimalControlProblem: new FormControl(false),
    });

    public get isOptimalControlProblem(): boolean {
        return this.formGroup.get('isOptimalControlProblem')!.value;
    }

    constructor(
        private readonly dialogRef: MatDialogRef<SimulationDialogComponent>
    ) {
        this.formGroup.get('costFunction')!.disable();
    }

    public ngOnInit(): void {
        this.formGroup
            .get('isOptimalControlProblem')!
            .valueChanges.pipe(
                tap((isOptimalControlProblem: boolean) => {
                    if (isOptimalControlProblem) {
                        this.formGroup.get('costFunction')!.enable();
                    } else {
                        this.formGroup.get('costFunction')!.disable();
                    }
                })
            )
            .subscribe();
    }

    public closeDialog(): void {
        this.dialogRef.close(null);
    }

    public simulate(): void {
        this.dialogRef.close(this.formGroup.value);
    }
}
