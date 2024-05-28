import { Component, Input, ViewChild } from '@angular/core';
import { InputType, RowScheme } from '@core/types/datatable.types';
import { IValues } from '@core/types/processing';
import { OptimalControlData } from '@core/types/run.types';
import { valuesToRowData } from '@core/utils';
import { IOutputData, SplitComponent } from 'angular-split';

@Component({
    selector: 'app-optimal-control-info-panel',
    templateUrl: './optimal-control-info-panel.component.html',
    styleUrls: ['./optimal-control-info-panel.component.scss'],
})
export class OptimalControlInfoPanelComponent {
    @ViewChild(SplitComponent) private readonly splitComponent!: SplitComponent;

    private _data: OptimalControlData | null = null;

    @Input() public set data(data: OptimalControlData | null) {
        this._data = data;

        this.onDataUpdate(this._data);
    }
    public get data(): OptimalControlData | null {
        return this._data;
    }

    public interventionsData: Record<string, number>[] = [];

    public interventionsRowScheme: RowScheme = {};

    public onGutterDBClick(event: IOutputData): void {
        this.splitComponent.setVisibleAreaSizes([50, 50]);
    }

    private onDataUpdate(data: OptimalControlData | null): void {
        if (!data) {
            this.interventionsRowScheme = {};
            this.interventionsData = [];

            return;
        }

        this.interventionsRowScheme = data.result[1].interventions.reduce(
            (rowScheme: RowScheme, intervention: IValues): RowScheme => ({
                ...rowScheme,
                [intervention.name]: {
                    name: intervention.name,
                    type: InputType.Number,
                },
            }),
            {}
        );

        this.interventionsData = valuesToRowData(data.result[1].interventions);
    }
}
