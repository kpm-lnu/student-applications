import { Component, OnDestroy } from '@angular/core';
import {
    AsyncValidator,
    ControlValueAccessor,
    FormControl,
    FormControlStatus,
    FormGroup,
    NG_ASYNC_VALIDATORS,
    NG_VALUE_ACCESSOR,
    ValidationErrors,
    Validators,
} from '@angular/forms';
import {
    ApproximationType,
    IOptimalControlParameters,
} from '@core/types/processing';
import { OnChangeFn, OnTouchedFn } from '@core/types/utils.types';
import { Observable, Subscription, first, map } from 'rxjs';

@Component({
    selector: 'app-optimal-control-parameters-input-panel',
    templateUrl: './optimal-control-parameters-input-panel.component.html',
    styleUrls: ['./optimal-control-parameters-input-panel.component.scss'],
    providers: [
        {
            provide: NG_VALUE_ACCESSOR,
            useExisting: OptimalControlParametersInputPanelComponent,
            multi: true,
        },
        {
            provide: NG_ASYNC_VALIDATORS,
            useExisting: OptimalControlParametersInputPanelComponent,
            multi: true,
        },
    ],
})
export class OptimalControlParametersInputPanelComponent
    implements ControlValueAccessor, AsyncValidator, OnDestroy
{
    private readonly subscription: Subscription = new Subscription();

    public readonly approximationTypes: {
        text: string;
        value: ApproximationType;
    }[] = [
        {
            text: 'Piecewise constant',
            value: ApproximationType.PiecewiseConstant,
        },
        {
            text: 'Piecewise linear',
            value: ApproximationType.PiecewiseLinear,
        },
    ];

    public readonly control: FormGroup = new FormGroup({
        time: new FormControl<number | null>(null, [Validators.required]),
        nodesAmount: new FormControl<number | null>(null, [
            Validators.required,
        ]),
        objectiveFunction: new FormControl<string | null>(null, [
            Validators.required,
        ]),
        intervention: new FormGroup({
            nodesAmount: new FormControl<number | null>(null, [
                Validators.required,
            ]),
            approximationType: new FormControl<ApproximationType | null>(
                this.approximationTypes[0].value,
                [Validators.required]
            ),
            lowerBoundary: new FormControl<number | null>(null, [
                Validators.required,
            ]),
            upperBoundary: new FormControl<number | null>(null, [
                Validators.required,
            ]),
        }),
    });

    public ngOnDestroy(): void {
        this.subscription.unsubscribe();
    }

    public writeValue(value: IOptimalControlParameters | null): void {
        this.control.setValue(
            value ?? {
                time: null,
                nodesAmount: null,
                objectiveFunction: null,
                intervention: {
                    nodesAmount: null,
                    approximationType: ApproximationType.PiecewiseConstant,
                    upperBoundary: null,
                    lowerBoundary: null,
                },
            }
        );
    }

    public registerOnChange(
        onChange: OnChangeFn<IOptimalControlParameters>
    ): void {
        this.subscription.add(this.control.valueChanges.subscribe(onChange));
    }

    public registerOnTouched(
        onTouched: OnTouchedFn<IOptimalControlParameters>
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

    public validate(): Observable<ValidationErrors | null> {
        return this.control.statusChanges.pipe(
            first((status: FormControlStatus): boolean => status !== 'PENDING'),
            map((status: FormControlStatus): ValidationErrors | null =>
                status === 'VALID'
                    ? null
                    : {
                          optimalControlParameters: true,
                      }
            )
        );
    }
}
