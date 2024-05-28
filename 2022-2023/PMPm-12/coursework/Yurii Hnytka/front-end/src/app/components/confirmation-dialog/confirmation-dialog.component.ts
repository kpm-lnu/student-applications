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
        private readonly dialogRef: MatDialogRef<ConfirmationDialogComponent>,
        @Inject(MAT_DIALOG_DATA)
        private readonly dialogData: {
            title?: string;
            message?: string;
            confirmText?: string;
            cancelText?: string;
        }
    ) {
        if (this.dialogData.title) {
            this.title = this.dialogData.title;
        }

        if (this.dialogData.message) {
            this.message = this.dialogData.message;
        }

        if (this.dialogData.confirmText) {
            this.confirmText = this.dialogData.confirmText;
        }

        if (this.dialogData.cancelText) {
            this.cancelText = this.dialogData.cancelText;
        }
    }

    public cancel(): void {
        this.dialogRef.close(false);
    }

    public confirm(): void {
        this.dialogRef.close(true);
    }
}
