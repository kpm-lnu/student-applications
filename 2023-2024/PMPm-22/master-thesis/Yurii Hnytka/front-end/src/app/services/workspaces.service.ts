import { Injectable } from '@angular/core';
import { ModelDefinition } from '@core/classes/model.class';
import { IModel } from '@core/types/model.types';
import { IWorkspace } from '@core/types/workspaces.types';
import { BehaviorSubject, Observable, map } from 'rxjs';
import { ModelService } from './model.service';
import { SamplesService } from './samples.service';

const emptyWorkspace: IWorkspace = {
    name: '',
    model: new ModelDefinition(),
};

@Injectable({
    providedIn: 'root',
})
export class WorkspacesService {
    private _workspaces: IWorkspace[] = [];
    private readonly _currentSubject: BehaviorSubject<IWorkspace> =
        new BehaviorSubject<IWorkspace>(emptyWorkspace);
    private get _current(): IWorkspace {
        return this._currentSubject.value;
    }

    public readonly current$: Observable<IWorkspace> =
        this._currentSubject.asObservable();

    public get workspaces(): IWorkspace[] {
        return structuredClone(this._workspaces);
    }
    public get current(): IWorkspace {
        return structuredClone(this._current);
    }
    public get names(): string[] {
        return this._workspaces.map(
            (workspace: IWorkspace): string => workspace.name
        );
    }

    public constructor(
        private readonly samplesService: SamplesService,
        private readonly modelService: ModelService
    ) {}

    public addNew(): void {
        const model: ModelDefinition = new ModelDefinition();

        const workspace: IWorkspace = {
            name: `Workspace ${this._workspaces.length + 1}`,
            model,
        };

        this._workspaces.push(workspace);
        this._currentSubject.next(workspace);
        this.modelService.set(model);
    }

    public addSample(name: string): Observable<void> {
        return this.samplesService.getSample<IModel>(name).pipe(
            map((definition: IModel): void => {
                const model: ModelDefinition = new ModelDefinition(definition);

                const workspace: IWorkspace = {
                    name: `Workspace ${this._workspaces.length + 1}`,
                    model,
                };

                this._workspaces.push(workspace);
                this._currentSubject.next(workspace);
                this.modelService.set(model);
            })
        );
    }

    public set(name: string): void {
        const workspace: IWorkspace = this._workspaces.find(
            (workspace: IWorkspace): boolean => workspace.name === name
        ) as IWorkspace;

        this._currentSubject.next(workspace);
        this.modelService.set(this._current.model);
    }

    public replaceModel(definition: IModel): void {
        const model: ModelDefinition = new ModelDefinition(definition);

        this._current.model = model;
        this.modelService.set(model);
    }

    public remove(): void {
        const index: number = this._workspaces.findIndex(
            (workspace: IWorkspace): boolean =>
                workspace.name === this._current.name
        );

        this._workspaces.splice(index, 1);
        this._currentSubject.next(
            this._workspaces[Math.max(index - 1, 0)] || emptyWorkspace
        );
        this.modelService.set(this._current.model);

        this.updateWorkspaceNames();
    }

    private updateWorkspaceNames(): void {
        this._workspaces.forEach(
            (workspace: IWorkspace, index: number): void => {
                workspace.name = `Workspace ${index + 1}`;
            }
        );
    }
}
