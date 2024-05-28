import { ValidatorFn } from '@angular/forms';
import { IOption } from './utils.types';

export interface IRow<DataType extends Record<string, any> = any> {
    data: DataType;
    editing: boolean;
}

export interface IColumnScheme {
    name: string;
    type: InputType;
    editable?: boolean;
    validationFns?: ValidatorFn[];
}

export interface ITextColumnScheme extends IColumnScheme {
    type: InputType.Text;
}

export interface INumberColumnScheme extends IColumnScheme {
    type: InputType.Number;
}

export interface ISelectColumnScheme extends IColumnScheme {
    type: InputType.Select;
    exclusive: boolean;
    options: IOption[];
}

export type ColumnScheme =
    | ITextColumnScheme
    | INumberColumnScheme
    | ISelectColumnScheme;

export type RowScheme<ValueType = any> = {
    [Key in Extract<keyof ValueType, string>]: ColumnScheme;
};

export enum InputType {
    Text = 'text',
    Number = 'number',
    Select = 'select',
}
