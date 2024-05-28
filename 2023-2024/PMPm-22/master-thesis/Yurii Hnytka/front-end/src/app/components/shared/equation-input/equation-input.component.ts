import {
    AfterViewInit,
    Component,
    ElementRef,
    EventEmitter,
    Input,
    OnDestroy,
    ViewChild,
} from '@angular/core';
import {
    AsyncValidator,
    ControlValueAccessor,
    FormControl,
    NG_ASYNC_VALIDATORS,
    NG_VALUE_ACCESSOR,
    ValidationErrors,
    Validators,
} from '@angular/forms';
import {
    ICompartment,
    IConstant,
    IIntervention,
} from '@core/types/model.types';
import { IValidationResponse } from '@core/types/processing';
import { OnChangeFn, OnTouchedFn } from '@core/types/utils.types';
import {
    Observable,
    Subject,
    Subscription,
    debounceTime,
    distinctUntilChanged,
    filter,
    switchMap,
    take,
    tap,
} from 'rxjs';
import { ModelService } from 'src/app/services/model.service';
import { ValidationService } from 'src/app/services/validation.service';

@Component({
    selector: 'app-equation-input',
    templateUrl: './equation-input.component.html',
    styleUrls: ['./equation-input.component.scss'],
    providers: [
        {
            provide: NG_VALUE_ACCESSOR,
            useExisting: EquationInputComponent,
            multi: true,
        },
        {
            provide: NG_ASYNC_VALIDATORS,
            useExisting: EquationInputComponent,
            multi: true,
        },
    ],
})
export class EquationInputComponent
    implements ControlValueAccessor, AfterViewInit, OnDestroy, AsyncValidator
{
    @ViewChild('input') private equationInput!: ElementRef;

    private readonly validationRequest: Subject<string> = new Subject();
    private readonly validationResult: EventEmitter<ValidationErrors | null> =
        new EventEmitter<ValidationErrors | null>();
    private readonly subscription: Subscription = new Subscription();

    @Input() public title: string = '';
    @Input() public placeholder: string = '';

    public readonly control: FormControl<string> = new FormControl<string>('', [
        Validators.required,
    ]) as FormControl<string>;

    public readonly compartments: ICompartment[] =
        this.modelService.compartments;
    public readonly constants: IConstant[] = this.modelService.constants;
    public readonly interventions: IIntervention[] =
        this.modelService.interventions;

    constructor(
        public readonly modelService: ModelService,
        private readonly validationService: ValidationService
    ) {}

    public ngAfterViewInit(): void {
        const inputLength: number = this.control.value
            ? this.control.value.length
            : 0;

        this.setCursorPosition(inputLength);
        this.initValidator();

        this.validationRequest.next(this.control.value);
    }

    public ngOnDestroy(): void {
        this.subscription.unsubscribe();
    }

    public onChipInput(name: string): void {
        const equation: string = this.control.value;
        const cursorPosition: number = this.getCursorPosition();

        const leftPart: string = equation
            ? equation.slice(0, cursorPosition).trim()
            : '';
        const rightPart: string = equation
            ? equation.slice(cursorPosition).trim()
            : '';

        this.control.setValue(`${leftPart} ${name} ${rightPart}`);

        this.setCursorPosition(leftPart.length + name.length + 2);
    }

    public writeValue(equation: string): void {
        this.control.setValue(equation);
    }

    public registerOnChange(onChange: OnChangeFn<string | null>): void {
        this.subscription.add(this.control.valueChanges.subscribe(onChange));
    }

    public registerOnTouched(onTouched: OnTouchedFn<string | null>): void {
        this.subscription.add(this.control.valueChanges.subscribe(onTouched));
    }

    public setDisabledState(isDisabled: boolean): void {
        if (isDisabled) {
            this.control.disable();

            return;
        }

        this.control.enable();
    }

    public validate(): Observable<ValidationErrors | null> {
        this.validationRequest.next(this.control.value);

        return this.validationResult.pipe(take(1));
    }

    private initValidator(): void {
        const validationSubscription: Subscription = this.validationRequest
            .pipe(
                filter((equation: string): boolean => !!equation),
                debounceTime(500),
                distinctUntilChanged(),
                switchMap(
                    (equation: string): Observable<IValidationResponse> =>
                        this.validationService.expression(
                            equation,
                            this.modelService.symbols
                        )
                ),
                tap((response: IValidationResponse): void => {
                    if (!response.valid) {
                        const errors: ValidationErrors = {
                            equation: {
                                invalid: response.message,
                            },
                        };

                        this.control.setErrors(errors);
                        this.validationResult.next(errors);

                        return;
                    }

                    this.control.setErrors(null);
                    this.validationResult.next(null);
                })
            )
            .subscribe();

        this.subscription.add(validationSubscription);
    }

    private setCursorPosition(position: number): void {
        this.equationInput.nativeElement.focus();

        this.equationInput.nativeElement.selectionStart = position;
        this.equationInput.nativeElement.selectionEnd = position;
    }

    private getCursorPosition(): number {
        return this.equationInput.nativeElement.selectionStart || 0;
    }
}
