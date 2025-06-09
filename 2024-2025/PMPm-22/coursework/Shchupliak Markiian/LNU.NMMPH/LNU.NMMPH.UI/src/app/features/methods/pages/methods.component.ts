import { Component, OnInit, ViewChild } from '@angular/core';
import { IMethod } from '../models/method';
import { MethodsService } from '../services/methods.service';
import { MessageService } from 'primeng/api';
import { FileUpload } from 'primeng/fileupload';
import { saveAs } from 'file-saver';

@Component({
  selector: 'app-methods',
  templateUrl: './methods.component.html',
  styleUrls: ['./methods.component.scss']
})
export class MethodsComponent implements OnInit {
  Euler: string = 'euler_method.csx';
  RungeKutta: string = 'rk_method.csx';

  methods: IMethod[] | undefined;
  file: any | null = null;
  result: number | null = null;

  selectedMethod: IMethod | null = null;

  @ViewChild('fileUploadControl') fileUploadControl!: FileUpload;

  constructor(private methodsService: MethodsService, private messageService: MessageService) { }

  ngOnInit() {
    this.methods = [
      { name: 'Euler', id: 1 },
      { name: 'Runge-Kutta', id: 2 },
    ];
  }

  downloadTemplate() {
    let fileName = '';

    if (this.selectedMethod?.id === 1) {
      fileName = this.Euler;
    } else if (this.selectedMethod?.id === 2) {
      fileName = this.RungeKutta;
    }

    this.methodsService.getTemplate(fileName)
      .subscribe({
        next: (response: Blob) => {
          this.saveFile(response, fileName);
        },
        error: (err) => {
          console.error('Download failed', err);
        }
      });
  }

  saveFile(blob: Blob, fileName: string): void {
    saveAs(blob, fileName);
  }

  onUpload(event: any) {
    const formData = new FormData();
    formData.append('file', event.files[0], event.files[0].name);
    formData.append('method', this.selectedMethod?.id.toString() || '');

    this.methodsService.postFile(formData)
      .subscribe({
        next: (response: number) => {
          this.messageService.add({ severity: 'info', summary: 'Success', detail: 'File Uploaded' });
          this.result = response;
        }
      });
  }

  onChange(event: any) {
    this.result = null;
    this.file = null;
    this.fileUploadControl.clear();
  }
}
