import { Component, OnDestroy } from '@angular/core';
import {
    ControlValueAccessor,
    FormControl,
    FormGroup,
    NG_VALIDATORS,
    NG_VALUE_ACCESSOR,
    ValidationErrors,
    Validator,
    Validators,
} from '@angular/forms';
import { ISimulationParameters } from '@core/types/processing';
import { OnChangeFn, OnTouchedFn } from '@core/types/utils.types';
import { Subscription } from 'rxjs';

@Component({
    selector: 'app-simulation-parameters-input-panel',
    templateUrl: './simulation-parameters-input-panel.component.html',
    styleUrls: ['./simulation-parameters-input-panel.component.scss'],
    providers: [
        {
            provide: NG_VALUE_ACCESSOR,
            useExisting: SimulationParametersInputPanelComponent,
            multi: true,
        },
        {
            provide: NG_VALIDATORS,
            useExisting: SimulationParametersInputPanelComponent,
            multi: true,
        },
    ],
})
export class SimulationParametersInputPanelComponent
    implements ControlValueAccessor, Validator, OnDestroy
{
    private readonly subscription: Subscription = new Subscription();

    public readonly control: FormGroup = new FormGroup({
        time: new FormControl<number | null>(null, [
            Validators.required,
            Validators.min(0 + Number.EPSILON),
        ]),
        nodesAmount: new FormControl<number | null>(null, [
            Validators.required,
            Validators.min(1),
        ]),
    });

    public ngOnDestroy(): void {
        this.subscription.unsubscribe();
    }

    public writeValue(value: ISimulationParameters | null): void {
        this.control.setValue(
            value ?? {
                time: null,
                nodesAmount: null,
            }
        );
    }

    public registerOnChange(onChange: OnChangeFn<ISimulationParameters>): void {
        this.subscription.add(this.control.valueChanges.subscribe(onChange));
    }

    public registerOnTouched(
        onTouched: OnTouchedFn<ISimulationParameters>
    ): void {
        this.subscription.add(this.control.valueChanges.subscribe(onTouched));
    }

    public setDisabledState(disabled: boolean): void {
        if (disabled) {
            this.control.disable();

            return;
        }

        this.control.enable();
    }

    public validate(): ValidationErrors | null {
        return this.control.valid ? null : { simulationParameters: true };
    }
}
