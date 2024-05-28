import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { AbstractControl, ValidationErrors, ValidatorFn } from '@angular/forms';
import { ICompartment } from '@core/types/model.types';
import { IValidationResponse } from '@core/types/processing';
import { Observable } from 'rxjs';
import { ModelService } from './model.service';

@Injectable({
    providedIn: 'root',
})
export class ValidationService {
    constructor(
        private readonly httpClient: HttpClient,
        private readonly modelService: ModelService
    ) {}

    public expression(
        expression: string,
        allowedSymbols: string[]
    ): Observable<IValidationResponse> {
        return this.httpClient.post<IValidationResponse>(
            'http://localhost:5000/validate-expression',
            {
                expression,
                allowedSymbols,
            }
        );
    }

    public getDefinitionNameValidator(initialName?: string): ValidatorFn {
        return (control: AbstractControl): ValidationErrors | null => {
            const name: any = control.value;

            if (!name || typeof name !== 'string' || name === initialName) {
                return null;
            }

            return this.definitionName(control.value);
        };
    }

    public definitionName(name: string): ValidationErrors | null {
        if (!name) {
            return null;
        }

        if (!this.name(name)) {
            return { invalidName: true };
        }

        return this.modelService.compartments.some(
            (compartment: ICompartment): boolean => compartment.name === name
        )
            ? { nameExists: true }
            : null;
    }

    private name(name: string): boolean {
        return /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name);
    }
}
