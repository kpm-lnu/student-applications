import {
    Component,
    Input,
    ViewChild
} from '@angular/core';
import { InputType, RowScheme } from '@core/types/datatable.types';
import {
    IdentifiedConstantDefinition,
    SelectedConstantDefinition,
} from '@core/types/definitions.types';
import { IValues } from '@core/types/processing';
import { PIData } from '@core/types/run.types';
import { valuesToRowData } from '@core/utils';
import { IOutputData, SplitComponent } from 'angular-split';

@Component({
    selector: 'app-parameters-identification-info-panel',
    templateUrl: './parameters-identification-info-panel.component.html',
    styleUrls: ['./parameters-identification-info-panel.component.scss'],
})
export class ParametersIdentificationInfoPanelComponent {
    @ViewChild(SplitComponent) private readonly splitComponent!: SplitComponent;

    private _data: PIData | null = null;

    @Input() public set data(data: PIData | null) {
        this._data = data;

        this.onDataUpdate(this._data);
    }
    public get data(): PIData | null {
        return this._data;
    }

    public identifiedConstantsData: IdentifiedConstantDefinition[] = [];
    public selectedConstantsData: SelectedConstantDefinition[] = [];
    public dataData: Record<string, number>[] = [];

    public dataRowScheme: RowScheme = {};

    public readonly selectedConstantsRowScheme: RowScheme<SelectedConstantDefinition> =
        {
            name: {
                name: 'Name',
                type: InputType.Text,
            },
            lowerBoundary: {
                name: 'Lower boundary',
                type: InputType.Number,
            },
            value: {
                name: 'Initial guess',
                type: InputType.Number,
            },
            upperBoundary: {
                name: 'Upper boundary',
                type: InputType.Number,
            },
        };

    public readonly identifiedConstantsRowScheme: RowScheme<IdentifiedConstantDefinition> =
        {
            name: {
                name: 'Name',
                type: InputType.Text,
            },
            value: {
                name: 'Value',
                type: InputType.Number,
            },
        };

    public onGutterDBClick(event: IOutputData): void {
        this.splitComponent.setVisibleAreaSizes([50, 50]);
    }

    public onDataUpdate(data: PIData | null): void {
        if (!data) {
            this.dataRowScheme = {};

            this.identifiedConstantsData = [];
            this.selectedConstantsData = [];
            this.dataData = [];

            return;
        }

        this.dataRowScheme = data.parameters.data.reduce(
            (rowScheme: RowScheme, values: IValues): RowScheme => ({
                ...rowScheme,
                [values.name]: {
                    name: values.name,
                    type: InputType.Number,
                },
            }),
            {}
        );

        this.identifiedConstantsData = data.result.constants;
        this.selectedConstantsData = data.parameters.selectedConstants;
        this.dataData = valuesToRowData(data.parameters.data);
    }
}
