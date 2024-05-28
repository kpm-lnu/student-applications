import { HttpClientModule } from '@angular/common/http';
import { NgModule } from '@angular/core';
import { ReactiveFormsModule } from '@angular/forms';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { MatOptionModule } from '@angular/material/core';
import { MatDialogModule } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatTableModule } from '@angular/material/table';

import { PlotlyModule } from 'angular-plotly.js';
import PlotlyJS from 'plotly.js-dist-min';

import { AppComponent } from './app.component';
import { CompartmentCreationDialogComponent } from './components/compartment-creation-dialog/compartment-creation-dialog.component';
import { ConfirmationDialogComponent } from './components/confirmation-dialog/confirmation-dialog.component';
import { DefinitionsTableDialogComponent } from './components/definitions-table-dialog/definitions-table-dialog.component';
import { EquationInputComponent } from './components/equation-input/equation-input.component';
import { FlowCreationDialogComponent } from './components/flow-creation-dialog/flow-creation-dialog.component';
import { SimulationDashboardComponent } from './components/simulation-dashboard/simulation-dashboard.component';
import { SimulationDialogComponent } from './components/simulation-dialog/simulation-dialog.component';

PlotlyModule.plotlyjs = PlotlyJS;

@NgModule({
    declarations: [
        AppComponent,
        CompartmentCreationDialogComponent,
        ConfirmationDialogComponent,
        SimulationDialogComponent,
        FlowCreationDialogComponent,
        SimulationDashboardComponent,
        DefinitionsTableDialogComponent,
        EquationInputComponent,
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
    ],
    providers: [],
    bootstrap: [AppComponent],
})
export class AppModule {}
