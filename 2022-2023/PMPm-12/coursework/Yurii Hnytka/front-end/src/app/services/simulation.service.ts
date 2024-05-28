import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { IOptimalControlData, ISimulationData, ISimulationResults } from '../core/interfaces';

@Injectable({
    providedIn: 'root',
})
export class SimulationService {
    constructor(private readonly httpClient: HttpClient) {}

    public simulateModel(
        model: ISimulationData
    ): Observable<ISimulationResults> {
        return this.httpClient.post<ISimulationResults>(
            'http://localhost:5000/simulate',
            model
        );
    }

    public optimizeModel(
        model: IOptimalControlData
    ): Observable<ISimulationResults> {
        return this.httpClient.post<ISimulationResults>(
            'http://localhost:5000/optimalControl',
            model
        );
    }
}
