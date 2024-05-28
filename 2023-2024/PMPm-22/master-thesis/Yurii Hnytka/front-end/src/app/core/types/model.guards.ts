import {
    ICompartment,
    IConstant,
    IFlow,
    IIntervention,
    IModel,
} from './model.types';

export function isCompartment(compartment: any): compartment is ICompartment {
    const isKeysAmountValid: boolean = Object.keys(compartment).length === 3;

    const isIdValid: boolean =
        'id' in compartment && typeof compartment.id === 'string';
    const isNameValid: boolean =
        'name' in compartment && typeof compartment.name === 'string';
    const isValueValid: boolean =
        'value' in compartment && typeof compartment.value === 'number';

    return isKeysAmountValid && isIdValid && isNameValid && isValueValid;
}

export function isConstant(constant: any): constant is IConstant {
    const isKeysAmountValid: boolean = Object.keys(constant).length === 3;

    const isIdValid: boolean =
        'id' in constant && typeof constant.id === 'string';
    const isNameValid: boolean =
        'name' in constant && typeof constant.name === 'string';
    const isValueValid: boolean =
        'value' in constant && typeof constant.value === 'number';

    return isKeysAmountValid && isIdValid && isNameValid && isValueValid;
}

export function isIntervention(
    intervention: any
): intervention is IIntervention {
    const isKeysAmountValid: boolean = Object.keys(intervention).length === 2;

    const isIdValid: boolean =
        'id' in intervention && typeof intervention.id === 'string';
    const isNameValid: boolean =
        'name' in intervention && typeof intervention.name === 'string';

    return isKeysAmountValid && isIdValid && isNameValid;
}

export function isFlow(flow: any): flow is IFlow {
    const isKeysAmountValid: boolean = Object.keys(flow).length === 4;

    const isIdValid: boolean = 'id' in flow && typeof flow.id === 'string';
    const isSourceValid: boolean =
        'source' in flow && typeof flow.source === 'string';
    const isTargetValid: boolean =
        'target' in flow && typeof flow.target === 'string';
    const isEquationValid: boolean =
        'equation' in flow && typeof flow.equation === 'string';

    return (
        isKeysAmountValid &&
        isIdValid &&
        isSourceValid &&
        isTargetValid &&
        isEquationValid
    );
}

export function isModel(model: any): model is IModel {
    const isKeysAmountValid: boolean = Object.keys(model).length === 4;

    const isCompartmentsValid: boolean =
        'compartments' in model &&
        Array.isArray(model.compartments) &&
        model.compartments.every((compartment: any): boolean =>
            isCompartment(compartment)
        );
    const isConstantsValid: boolean =
        'constants' in model &&
        Array.isArray(model.constants) &&
        model.constants.every((constant: any): boolean => isConstant(constant));
    const isInterventionsValid: boolean =
        'interventions' in model &&
        Array.isArray(model.interventions) &&
        model.interventions.every((intervention: any): boolean =>
            isIntervention(intervention)
        );
    const isFlowsValid: boolean =
        'flows' in model &&
        Array.isArray(model.flows) &&
        model.flows.every((flow: any): boolean => isFlow(flow));

    return (
        isKeysAmountValid &&
        isCompartmentsValid &&
        isConstantsValid &&
        isInterventionsValid &&
        isFlowsValid
    );
}
