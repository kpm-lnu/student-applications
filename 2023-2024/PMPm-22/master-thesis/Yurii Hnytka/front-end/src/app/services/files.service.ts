import { Injectable } from '@angular/core';
import { Observable, Observer } from 'rxjs';

@Injectable({
    providedIn: 'root',
})
export class FilesService {
    public downloadFileWithData(data: any, filename: string): void {
        const dataString: string = JSON.stringify(data, null, 4);

        const blob: Blob = new Blob([dataString], {
            type: 'application/json',
        });

        const url: string = URL.createObjectURL(blob);

        const anchor: HTMLAnchorElement = document.createElement('a');
        anchor.href = url;
        anchor.download = filename;

        anchor.click();

        URL.revokeObjectURL(url);
        anchor.remove();
    }

    public readDataFromFile<DataType = any>(
        fileExtension: string
    ): Observable<DataType> {
        return new Observable<DataType>(
            (observer: Observer<DataType>): void => {
                const fileInput: HTMLInputElement =
                    document.createElement('input');
                fileInput.type = 'file';
                fileInput.accept = fileExtension;

                fileInput.onchange = (): void => {
                    const file: File = fileInput.files![0];
                    const fileReader: FileReader = new FileReader();

                    fileReader.onload = (): void => {
                        const data: DataType = JSON.parse(
                            fileReader.result as string
                        );

                        observer.next(data);
                        observer.complete();
                    };

                    fileReader.readAsText(file);
                };

                fileInput.click();
                fileInput.remove();
            }
        );
    }
}
