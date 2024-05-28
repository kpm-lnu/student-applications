import { Component, Input } from '@angular/core';
import { SimulationData } from '@core/types/run.types';

@Component({
    selector: 'app-simulation-info-panel',
    templateUrl: './simulation-info-panel.component.html',
    styleUrls: ['./simulation-info-panel.component.scss'],
})
export class SimulationInfoPanelComponent {
    @Input() public data: SimulationData | null = null;
}
