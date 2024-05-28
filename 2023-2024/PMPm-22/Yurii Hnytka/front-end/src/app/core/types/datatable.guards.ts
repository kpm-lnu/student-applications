import {
    ColumnScheme,
    IColumnScheme,
    INumberColumnScheme,
    IRow,
    ISelectColumnScheme,
    ITextColumnScheme,
    InputType,
    RowScheme,
} from './datatable.types';
import { isOption } from './utils.guards';

export function isRow(row: any): row is IRow {
    const isKeysAmountValid: boolean = Object.keys(row).length === 2;

    const isDataValid: boolean = 'data' in row;
    const isEditingValid: boolean =
        'editing' in row && typeof row.editing === 'boolean';

    return isKeysAmountValid && isDataValid && isEditingValid;
}

export function isIColumnScheme(
    iColumnScheme: any
): iColumnScheme is IColumnScheme {
    const isKeysAmountValid: boolean = Object.keys(iColumnScheme).length <= 4;

    const isNameValid: boolean =
        'name' in iColumnScheme && typeof iColumnScheme.name === 'string';
    const isInputTypeValid: boolean =
        'type' in iColumnScheme && isInputType(iColumnScheme.type);

    let isEditableValid: boolean = true;
    let isValidationFnsValid: boolean = true;

    if ('editable' in iColumnScheme) {
        isEditableValid = typeof iColumnScheme.editable === 'boolean';
    }

    if ('validationFns' in iColumnScheme) {
        isValidationFnsValid =
            Array.isArray(iColumnScheme.validationFns) &&
            iColumnScheme.validationFns.every(
                (validationFn: any): boolean =>
                    typeof validationFn === 'function'
            );
    }

    return (
        isKeysAmountValid &&
        isNameValid &&
        isInputTypeValid &&
        isEditableValid &&
        isValidationFnsValid
    );
}

export function isTextColumnScheme(
    textColumnScheme: any
): textColumnScheme is ITextColumnScheme {
    const isIColumnSchemeValid: boolean = isIColumnScheme(textColumnScheme);

    const isTypeValid: boolean = textColumnScheme.inputType === InputType.Text;

    return isIColumnSchemeValid && isTypeValid;
}

export function isNumberColumnScheme(
    numberColumnScheme: any
): numberColumnScheme is INumberColumnScheme {
    const isIColumnSchemeValid: boolean = isIColumnScheme(numberColumnScheme);

    const isTypeValid: boolean =
        numberColumnScheme.inputType === InputType.Number;

    return isIColumnSchemeValid && isTypeValid;
}

export function isSelectColumnScheme(
    selectColumnScheme: any
): selectColumnScheme is ISelectColumnScheme {
    const isIColumnSchemeValid: boolean = isIColumnScheme(selectColumnScheme);

    const isTypeValid: boolean =
        selectColumnScheme.inputType === InputType.Select;
    const isExclusiveValid: boolean =
        'exclusive' in selectColumnScheme &&
        typeof selectColumnScheme.exclusive === 'boolean';
    const isOptionsValid: boolean =
        'options' in selectColumnScheme &&
        Array.isArray(selectColumnScheme.options) &&
        selectColumnScheme.options.every((option: any): boolean =>
            isOption(option)
        );

    return (
        isIColumnSchemeValid &&
        isTypeValid &&
        isExclusiveValid &&
        isOptionsValid
    );
}

export function isColumnScheme(
    columnScheme: any
): columnScheme is ColumnScheme {
    return (
        isTextColumnScheme(columnScheme) ||
        isNumberColumnScheme(columnScheme) ||
        isSelectColumnScheme(columnScheme)
    );
}

export function isRowScheme(rowScheme: any): rowScheme is RowScheme {
    const isObjectValid: boolean = typeof rowScheme === 'object';
    const areKeysValid: boolean = Object.entries(rowScheme).every(
        ([key, columnScheme]: [string, any]): boolean =>
            typeof key === 'string' && isColumnScheme(columnScheme)
    );

    return isObjectValid && areKeysValid;
}

export function isInputType(inputType: any): inputType is InputType {
    return Object.values(InputType).includes(inputType);
}
