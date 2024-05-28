
export type OnChangeFn<ValueType = any> = (value: ValueType) => void;
export type OnTouchedFn<ValueType = any> = (value: ValueType) => void;

export type Nullable<Type> = {
    [Property in keyof Type]: Type[Property] | null;
};

export interface IOption {
    value: string;
    label: string;
}
