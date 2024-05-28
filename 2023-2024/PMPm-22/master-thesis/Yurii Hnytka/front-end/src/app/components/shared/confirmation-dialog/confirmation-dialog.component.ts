import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';

@Component({
    selector: 'app-confirmation-dialog',
    templateUrl: './confirmation-dialog.component.html',
    styleUrls: ['./confirmation-dialog.component.scss'],
})
export class ConfirmationDialogComponent {
    public readonly title: string = 'Confirmation';
    public readonly message: string = 'Are you sure you want to do this?';
    public readonly confirmText: string = 'Confirm';
    public readonly cancelText: string = 'Cancel';

    constructor(
        private readonly dialogRef: MatDialogRef<
            ConfirmationDialogComponent,
            boolean
        >,
        @Inject(MAT_DIALOG_DATA)
        private readonly data: {
            title?: string;
            message?: string;
            confirmText?: string;
            cancelText?: string;
        }
    ) {
        if (this.data.title) {
            this.title = this.data.title;
        }

        if (this.data.message) {
            this.message = this.data.message;
        }

        if (this.data.confirmText) {
            this.confirmText = this.data.confirmText;
        }

        if (this.data.cancelText) {
            this.cancelText = this.data.cancelText;
        }
    }

    public close(): void {
        this.dialogRef.close();
    }

    public cancel(): void {
        this.dialogRef.close(false);
    }

    public confirm(): void {
        this.dialogRef.close(true);
    }
}
