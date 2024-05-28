import { DataSource } from '@angular/cdk/collections';
import { Injectable } from '@angular/core';
import {
    AbstractControl,
    FormArray,
    FormControl,
    FormGroup,
    Validators,
} from '@angular/forms';
import { BehaviorSubject, Observable, Subscription, tap } from 'rxjs';
import { Definition, IDefinitionsTable } from '../core/interfaces';

@Injectable({
    providedIn: 'root',
})
export class DefinitionsService {
    private readonly definitionsFormArray: FormArray;

    private readonly definitionsFormArrayControlsSubject: BehaviorSubject<
        FormArray['controls']
    >;

    private readonly subscription: Subscription = new Subscription();

    constructor() {
        this.definitionsFormArray = new FormArray([
            this.getNewDefinitionFormGroup(),
        ]);

        this.definitionsFormArrayControlsSubject = new BehaviorSubject<
            FormArray['controls']
        >(this.definitionsFormArray.controls);
    }

    public addDefinition(definition: Definition): void {
        const newDefinitionFormGroup: FormGroup =
            this.getNewDefinitionFormGroup();

        newDefinitionFormGroup.patchValue(definition);

        this.definitionsFormArray.insert(-1, newDefinitionFormGroup);

        this.definitionsFormArrayControlsSubject.next(
            this.definitionsFormArray.controls
        );
    }

    public removeDefinition(definitionId: string): void {
        const index: number = this.definitionsFormArray.controls.findIndex(
            (control: AbstractControl): boolean =>
                control.value.id === definitionId
        );

        if (index !== -1) {
            this.definitionsFormArray.removeAt(index);

            if (this.definitionsFormArray.length === 0) {
                this.definitionsFormArray.push(
                    this.getNewDefinitionFormGroup()
                );
            }

            this.definitionsFormArrayControlsSubject.next(
                this.definitionsFormArray.controls
            );
        }
    }

    public getDataSource(): DataSource<AbstractControl> {
        return {
            connect: (): Observable<AbstractControl[]> =>
                this.definitionsFormArrayControlsSubject.asObservable(),
            disconnect: (): void => {},
        };
    }

    public getDefinitionsTable(): IDefinitionsTable {
        const table: IDefinitionsTable = {
            compartments: [],
            interventions: [],
            constants: [],
        };

        this.definitionsFormArray.value.forEach(
            (definition: Definition): void => {
                switch (definition.type) {
                    case 'compartment':
                        table.compartments.push(definition);

                        break;

                    case 'intervention':
                        table.interventions.push(definition);

                        break;

                    case 'constant':
                        table.constants.push(definition);

                        break;
                }
            }
        );

        return table;
    }

    public getAvailableSymbols(): string[] {
        const availableSymbols: string[] = [];

        this.definitionsFormArray.value.forEach(
            (definition: Definition): void => {
                if (!definition.id) {
                    return;
                }

                availableSymbols.push(definition.id);
            }
        );

        return availableSymbols;
    }

    private getNewDefinitionFormGroup(): FormGroup {
        const newDefinitionFormGroup: FormGroup = new FormGroup({
            id: new FormControl(null, [Validators.required]),
            type: new FormControl(null, [Validators.required]),
            value: new FormControl(null, [Validators.required]),
        });

        const newElementSub: Subscription = newDefinitionFormGroup.valueChanges
            .pipe(
                tap((): void => {
                    const index: number =
                        this.definitionsFormArray.controls.findIndex(
                            (control: AbstractControl): boolean =>
                                control === newDefinitionFormGroup
                        );

                    if (index === this.definitionsFormArray.length - 1) {
                        this.definitionsFormArray.push(
                            this.getNewDefinitionFormGroup.apply(this)
                        );

                        this.definitionsFormArrayControlsSubject.next(
                            this.definitionsFormArray.controls
                        );
                    }
                })
            )
            .subscribe();

        this.subscription.add(newElementSub);

        return newDefinitionFormGroup;
    }
}
