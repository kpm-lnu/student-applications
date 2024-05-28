import { Observable, Subscriber } from 'rxjs';
import { IValues } from './types/processing';

export function fromResizeObserver(
    target: Element
): Observable<ResizeObserverEntry> {
    return new Observable(
        (subscriber: Subscriber<ResizeObserverEntry>): (() => void) => {
            const resizeObserver = new ResizeObserver(
                (entries: ResizeObserverEntry[]): void => {
                    entries.forEach((entry: ResizeObserverEntry): void => {
                        subscriber.next(entry);
                    });
                }
            );

            resizeObserver.observe(target);

            return (): void => {
                resizeObserver.unobserve(target);
                resizeObserver.disconnect();
            };
        }
    );
}

export function valuesToRowData(values: IValues[]): Record<string, number>[] {
    if (!values.length) {
        return [];
    }

    return Array.from(
        { length: values[0].values.length },
        (_: unknown, index: number): Record<string, number> =>
            values.reduce(
                (
                    row: Record<string, number>,
                    values: IValues
                ): Record<string, number> => ({
                    ...row,
                    [values.name]: values.values[index],
                }),
                {}
            )
    );
}

export function rowDataToValues(rowData: Record<string, number>[]): IValues[] {
    if (!rowData.length) {
        return [];
    }

    return Object.keys(rowData[0]).reduce(
        (values: IValues[], name: string): IValues[] => {
            const currentValues = rowData.map(
                (row: Record<string, number>): number => row[name]
            );

            if (currentValues.some((value: number): boolean => !value)) {
                return values;
            }

            values.push({
                name,
                values: currentValues,
            });

            return values;
        },
        []
    );
}
