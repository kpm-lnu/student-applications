import {
    isOptimalControlParameters,
    isOptimalControlResult,
} from './processing/optimal-control.guards';
import {
    isPIParameters,
    isPIResult,
} from './processing/parameters-identification.guards';
import {
    isSimulationParameters,
    isSimulationResult,
} from './processing/simulation.guards';
import {
    Data,
    DataType,
    IRun,
    NoneData,
    NoneRun,
    OptimalControlData,
    OptimalControlRun,
    PIData,
    PIRun,
    Run,
    SimulationData,
    SimulationRun,
} from './run.types';

export function isIRun(iRun: any): iRun is IRun {
    const isKeysAmountValid: boolean = Object.keys(iRun).length === 2;

    const isNameValid: boolean =
        'name' in iRun && typeof iRun.name === 'string';
    const isDataValid: boolean = 'data' in iRun;

    return isKeysAmountValid && isNameValid && isDataValid;
}

export function isSimulationData(
    simulationData: any
): simulationData is SimulationData {
    const isKeysAmountValid: boolean = Object.keys(simulationData).length === 3;

    const isParametersValid: boolean =
        'parameters' in simulationData &&
        isSimulationParameters(simulationData.parameters);
    const isResultValid: boolean =
        'result' in simulationData && isSimulationResult(simulationData.result);
    const isTypeValid: boolean =
        'type' in simulationData &&
        isDataType(simulationData.type) &&
        simulationData.type === DataType.Simulation;

    return (
        isKeysAmountValid && isParametersValid && isResultValid && isTypeValid
    );
}

export function isOptimalControlData(
    optimalControlData: any
): optimalControlData is OptimalControlData {
    const isKeysAmountValid: boolean =
        Object.keys(optimalControlData).length === 3;

    const isParametersValid: boolean =
        'parameters' in optimalControlData &&
        isOptimalControlParameters(optimalControlData.parameters);
    const isResultValid: boolean =
        'result' in optimalControlData &&
        optimalControlData.result.length === 2 &&
        isSimulationResult(optimalControlData.result[0]) &&
        isOptimalControlResult(optimalControlData.result[1]);
    const isTypeValid: boolean =
        'type' in optimalControlData &&
        isDataType(optimalControlData.type) &&
        optimalControlData.type === DataType.OptimalControl;

    return (
        isKeysAmountValid && isParametersValid && isResultValid && isTypeValid
    );
}

export function isPIData(piData: any): piData is PIData {
    const isKeysAmountValid: boolean = Object.keys(piData).length === 3;

    const isParametersValid: boolean =
        'parameters' in piData && isPIParameters(piData.parameters);
    const isResultValid: boolean =
        'result' in piData && isPIResult(piData.result);
    const isTypeValid: boolean =
        'type' in piData &&
        isDataType(piData.type) &&
        piData.type === DataType.PI;

    return (
        isKeysAmountValid && isParametersValid && isResultValid && isTypeValid
    );
}

export function isNoneData(noneData: any): noneData is NoneData {
    const isKeysAmountValid: boolean = Object.keys(noneData).length === 1;

    const isTypeValid: boolean =
        'type' in noneData &&
        isDataType(noneData.type) &&
        noneData.type === DataType.None;

    return isKeysAmountValid && isTypeValid;
}

export function isData(data: any): data is Data {
    return (
        isSimulationData(data) ||
        isOptimalControlData(data) ||
        isPIData(data) ||
        isNoneData(data)
    );
}

export function isSimulationRun(
    simulationRun: any
): simulationRun is SimulationRun {
    const isIRunValid: boolean = isIRun(simulationRun);

    const isDataValid: boolean = isSimulationData(simulationRun.data);

    return isIRunValid && isDataValid;
}

export function isOptimalControlRun(
    optimalControlRun: any
): optimalControlRun is OptimalControlRun {
    const isIRunValid: boolean = isIRun(optimalControlRun);

    const isDataValid: boolean = isOptimalControlData(optimalControlRun.data);

    return isIRunValid && isDataValid;
}

export function isPIRun(piRun: any): piRun is PIRun {
    const isIRunValid: boolean = isIRun(piRun);

    const isDataValid: boolean = isPIData(piRun.data);

    return isIRunValid && isDataValid;
}

export function isNoneRun(noneRun: any): noneRun is NoneRun {
    const isIRunValid: boolean = isIRun(noneRun);

    const isDataValid: boolean = isNoneData(noneRun.data);

    return isIRunValid && isDataValid;
}

export function isRun(run: any): run is Run {
    return (
        isSimulationRun(run) ||
        isOptimalControlRun(run) ||
        isPIRun(run) ||
        isNoneRun(run)
    );
}

export function isDataType(dataType: any): dataType is DataType {
    return Object.values(DataType).includes(dataType);
}
