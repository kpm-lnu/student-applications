import { ProcessingType } from './dialog.types';

export function isProcessingType(
    processingType: any
): processingType is ProcessingType {
    return Object.values(ProcessingType).includes(processingType);
}
