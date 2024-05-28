import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import {
    IImportModel
} from '../core/interfaces';

@Injectable({
    providedIn: 'root',
})
export class SamplesService {
    constructor(private readonly httpClient: HttpClient) {}

    public getSample(sampleName: string): Observable<IImportModel> {
        return this.httpClient.get<IImportModel>(
            `/assets/samples/${sampleName}.json`,
            {
                responseType: 'json',
            }
        );
    }
}
