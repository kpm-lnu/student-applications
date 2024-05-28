import { DataSource } from '@angular/cdk/collections';
import { Component } from '@angular/core';
import { AbstractControl } from '@angular/forms';
import { MatDialogRef } from '@angular/material/dialog';
import { DefinitionsService } from 'src/app/services/definitions.service';

@Component({
    selector: 'app-definitions-table-dialog',
    templateUrl: './definitions-table-dialog.component.html',
    styleUrls: ['./definitions-table-dialog.component.scss'],
})
export class DefinitionsTableDialogComponent {
    public readonly displayedColumns: string[] = [
        'id',
        'type',
        'value',
        'actions',
    ];
    public readonly definitionTypes: { text: string; value: string }[] = [
        { text: 'Compartment', value: 'compartment' },
        { text: 'Intervention', value: 'intervention' },
        { text: 'Constant', value: 'constant' },
    ];

    public readonly definitionsDataSource: DataSource<AbstractControl>;

    constructor(
        private readonly dialogRef: MatDialogRef<DefinitionsTableDialogComponent>,
        private readonly definitionsService: DefinitionsService
    ) {
        this.definitionsDataSource = this.definitionsService.getDataSource();
    }

    public closeDialog(): void {
        this.dialogRef.close();
    }

    public removeDefinition(definitionId: string): void {
        this.definitionsService.removeDefinition(definitionId);
    }
}
