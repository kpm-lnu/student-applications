import { ModelDefinition } from '@core/classes/model.class';
import { IWorkspace } from './workspaces.types';

export function isWorkspace(workspace: any): workspace is IWorkspace {
    const isKeysAmountValid: boolean = Object.keys(workspace).length === 2;

    const isNameValid: boolean =
        'name' in workspace && typeof workspace.name === 'string';
    const isModelValid: boolean =
        'model' in workspace && workspace.model instanceof ModelDefinition;

    return isKeysAmountValid && isNameValid && isModelValid;
}
