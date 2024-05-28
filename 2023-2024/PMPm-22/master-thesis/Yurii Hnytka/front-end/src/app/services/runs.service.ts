import { Injectable } from '@angular/core';
import {
    isOptimalControlDataDefinition,
    isPIDataDefinition,
    isSimulationDataDefinition,
} from '@core/types/definitions.guards';
import { DataDefinition } from '@core/types/definitions.types';
import { IValues } from '@core/types/processing';
import {
    isOptimalControlRun,
    isPIRun,
    isSimulationRun,
} from '@core/types/run.guards';
import { Data, DataType, Run } from '@core/types/run.types';
import { BehaviorSubject, Observable } from 'rxjs';

const emptyRun: Run = {
    name: '',
    data: {
        type: DataType.None,
    },
};

@Injectable({
    providedIn: 'root',
})
export class RunsService {
    private _runs: Run[] = [];
    private readonly _currentSubject: BehaviorSubject<Run> =
        new BehaviorSubject<Run>(emptyRun);
    private get _current(): Run {
        return this._currentSubject.value;
    }

    public readonly current$: Observable<Run> =
        this._currentSubject.asObservable();
    public get runs(): Run[] {
        return structuredClone(this._runs);
    }
    public get current(): Run {
        return structuredClone(this._current);
    }
    public get names(): string[] {
        return this._runs.map((run: Run): string => run.name);
    }

    public get data(): Data {
        return this.current.data;
    }
    public get values(): IValues[] {
        const current: Run = this.current;

        if (isSimulationRun(current)) {
            return current.data.result.compartments;
        } else if (isOptimalControlRun(current)) {
            return [
                ...current.data.result[1].compartments,
                ...current.data.result[1].approximatedInterventions,
            ];
        } else if (isPIRun(current)) {
            return current.data.result.approximation;
        }

        return [];
    }

    public add(data: DataDefinition, set: boolean = false): void {
        const newRun: Run = {
            name: `Run ${this._runs.length + 1}`,
            data: {
                ...data,
                type: this.getDataType(data),
            },
        } as Run;

        this._runs.push(newRun);

        if (set || this._runs.length === 1) {
            this._currentSubject.next(newRun);
        }
    }

    public set(name: string): void {
        const run: Run = this._runs.find(
            (run: Run): boolean => run.name === name
        ) as Run;

        this._currentSubject.next(run);
    }

    public remove(): void {
        const index: number = this._runs.findIndex(
            (run: Run): boolean => run.name === this._current!.name
        );

        this._runs.splice(index, 1);
        this._currentSubject.next(
            this._runs[Math.max(index - 1, 0)] || emptyRun
        );

        this.updateNames();
    }

    private updateNames(): void {
        this._runs.forEach((run: Run, index: number): void => {
            run.name = `Run ${index + 1}`;
        });
    }

    private getDataType(definition: DataDefinition): DataType {
        if (isSimulationDataDefinition(definition)) {
            return DataType.Simulation;
        } else if (isOptimalControlDataDefinition(definition)) {
            return DataType.OptimalControl;
        } else if (isPIDataDefinition(definition)) {
            return DataType.PI;
        }

        return DataType.None;
    }
}
