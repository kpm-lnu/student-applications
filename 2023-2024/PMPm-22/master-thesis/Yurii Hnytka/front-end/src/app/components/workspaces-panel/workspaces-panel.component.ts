import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormControl } from '@angular/forms';
import { IWorkspace } from '@core/types/workspaces.types';
import { Subscription, filter, map, tap } from 'rxjs';
import { WorkspacesService } from 'src/app/services/workspaces.service';

@Component({
    selector: 'app-workspaces-panel',
    templateUrl: './workspaces-panel.component.html',
    styleUrls: ['./workspaces-panel.component.scss'],
})
export class WorkspacesPanelComponent implements OnInit, OnDestroy {
    private readonly subscription: Subscription = new Subscription();

    public readonly control: FormControl<string> = new FormControl<string>(
        ''
    ) as FormControl<string>;

    public constructor(public readonly workspacesService: WorkspacesService) {}

    public ngOnInit(): void {
        const nameSub: Subscription = this.workspacesService.current$
            .pipe(
                filter((workspace: IWorkspace): boolean => !!workspace.name),
                map((workspace: IWorkspace): string => workspace.name),
                tap(this.setName.bind(this))
            )
            .subscribe();

        const changeSub: Subscription = this.control.valueChanges
            .pipe(tap(this.set.bind(this)))
            .subscribe();

        this.subscription.add(nameSub);
        this.subscription.add(changeSub);
    }

    public ngOnDestroy(): void {
        this.subscription.unsubscribe();
    }

    public add(): void {
        this.workspacesService.addNew();
    }

    public remove(): void {
        this.workspacesService.remove();
    }

    private setName(name: string): void {
        this.control.setValue(name, {
            emitEvent: false,
        });
    }

    private set(name: string): void {
        this.workspacesService.set(name);
    }
}
