import { HttpClientModule } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { ReactiveFormsModule } from '@angular/forms';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { MatAutocompleteModule } from '@angular/material/autocomplete';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatChipsModule } from '@angular/material/chips';
import { MatOptionModule } from '@angular/material/core';
import { MatDialogModule } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatSnackBarModule } from '@angular/material/snack-bar';
import { MatTableModule } from '@angular/material/table';
import { MatTabsModule } from '@angular/material/tabs';

import { PlotlyModule } from 'angular-plotly.js';
import PlotlyJS from 'plotly.js-dist-min';

import { AngularSplitModule } from 'angular-split';
import { AppComponent } from './app.component';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { OptimalControlInfoPanelComponent } from './components/dashboard/optimal-control-info-panel/optimal-control-info-panel.component';
import { ParametersIdentificationInfoPanelComponent } from './components/dashboard/parameters-identification-info-panel/parameters-identification-info-panel.component';
import { SimulationInfoPanelComponent } from './components/dashboard/simulation-info-panel/simulation-info-panel.component';
import { DefinitionsTableDialogComponent } from './components/definitions-table-dialog/definitions-table-dialog.component';
import { CompartmentDialogComponent } from './components/graph/compartment-dialog/compartment-dialog.component';
import { FlowDialogComponent } from './components/graph/flow-dialog/flow-dialog.component';
import { OptimalControlParametersInputPanelComponent } from './components/processing/optimal-control-parameters-input-panel/optimal-control-parameters-input-panel.component';
import { ParametersIdentificationParametersInputPanelComponent } from './components/processing/parameters-identification-parameters-input-panel/parameters-identification-parameters-input-panel.component';
import { ProcessingDialogComponent } from './components/processing/processing-dialog/processing-dialog.component';
import { SimulationParametersInputPanelComponent } from './components/processing/simulation-parameters-input-panel/simulation-parameters-input-panel.component';
import { ConfirmationDialogComponent } from './components/shared/confirmation-dialog/confirmation-dialog.component';
import { DatatableComponent } from './components/shared/datatable/datatable.component';
import { EquationInputComponent } from './components/shared/equation-input/equation-input.component';
import { WorkspacesPanelComponent } from './components/workspaces-panel/workspaces-panel.component';

PlotlyModule.plotlyjs = PlotlyJS;

@NgModule({
    declarations: [
        AppComponent,
        CompartmentDialogComponent,
        ConfirmationDialogComponent,
        ProcessingDialogComponent,
        SimulationParametersInputPanelComponent,
        OptimalControlParametersInputPanelComponent,
        ParametersIdentificationParametersInputPanelComponent,
        FlowDialogComponent,
        DashboardComponent,
        DefinitionsTableDialogComponent,
        EquationInputComponent,
        DatatableComponent,
        WorkspacesPanelComponent,
        SimulationInfoPanelComponent,
        OptimalControlInfoPanelComponent,
        ParametersIdentificationInfoPanelComponent,
    ],
    imports: [
        BrowserModule,
        BrowserAnimationsModule,
        MatIconModule,
        MatButtonModule,
        MatDialogModule,
        MatInputModule,
        ReactiveFormsModule,
        HttpClientModule,
        PlotlyModule,
        MatTableModule,
        MatSelectModule,
        MatOptionModule,
        MatChipsModule,
        MatSlideToggleModule,
        MatSnackBarModule,
        MatButtonToggleModule,
        MatTabsModule,
        MatAutocompleteModule,
        AngularSplitModule,
    ],
    providers: [],
    bootstrap: [AppComponent],
})
export class AppModule {}
