import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import {
    OptimalControlRequestBody,
    OptimalControlResponse,
    PIRequestBody,
    PIResponse,
    SimulationRequestBody,
    SimulationResponse,
} from '@core/types/processing';
import { Observable } from 'rxjs';

@Injectable({
    providedIn: 'root',
})
export class ProcessingService {
    constructor(private readonly httpClient: HttpClient) {}

    public simulateModel(
        data: SimulationRequestBody
    ): Observable<SimulationResponse> {
        return this.httpClient.post<SimulationResponse>(
            'http://localhost:5000/simulate',
            data
        );
    }

    public optimizeModel(
        data: OptimalControlRequestBody
    ): Observable<OptimalControlResponse> {
        return this.httpClient.post<OptimalControlResponse>(
            'http://localhost:5000/optimal-control',
            data
        );
    }

    public identifyParameters(data: PIRequestBody): Observable<PIResponse> {
        return this.httpClient.post<PIResponse>(
            'http://localhost:5000/parameters-identification',
            data
        );
    }
}
