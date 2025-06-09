import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ButtonModule } from 'primeng/button';
import { DialogModule } from 'primeng/dialog';
import { MultiSelectModule } from 'primeng/multiselect';
import { TableModule } from 'primeng/table';
import { InputSwitchModule } from 'primeng/inputswitch';
import { InputTextModule } from 'primeng/inputtext';
import { CheckboxModule } from 'primeng/checkbox';
import { TabViewModule } from 'primeng/tabview';
import { DropdownModule } from 'primeng/dropdown';
import { SplitterModule } from 'primeng/splitter';
import { MenuModule } from 'primeng/menu';
import { ConfirmDialogModule } from 'primeng/confirmdialog';
import { MethodsRoutingModule } from './methods-routing.module';
import { MethodsComponent } from './pages/methods.component';
import { FileUploadModule } from 'primeng/fileupload';
import { provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';

@NgModule({
  declarations: [
    MethodsComponent
  ],
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,

    FileUploadModule,

    ButtonModule,
    DialogModule,
    MultiSelectModule,
    TableModule,
    InputSwitchModule,
    InputTextModule,
    CheckboxModule,
    DropdownModule,
    TabViewModule,
    SplitterModule,
    CheckboxModule,
    MenuModule,
    ConfirmDialogModule,

    MethodsRoutingModule
  ],
  exports: [
    MethodsComponent
  ],
  providers: [

  ]
})
export class MethodsModule { }
