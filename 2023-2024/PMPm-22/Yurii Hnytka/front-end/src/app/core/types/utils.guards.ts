import { IOption } from './utils.types';

export function isOption(option: any): option is IOption {
    const isKeysAmountValid: boolean = Object.keys(option).length === 2;

    const isValueValid: boolean =
        'value' in option && typeof option.value === 'string';
    const isLabelValid: boolean =
        'label' in option && typeof option.label === 'string';

    return isKeysAmountValid && isValueValid && isLabelValid;
}
