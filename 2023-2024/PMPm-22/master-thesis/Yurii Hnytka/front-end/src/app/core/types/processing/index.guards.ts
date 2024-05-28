import { Parameters } from './index';
import { isOptimalControlParameters } from './optimal-control.guards';
import { isPIParameters } from './parameters-identification.guards';
import { isSimulationParameters } from './simulation.guards';

export function isParameters(parameters: any): parameters is Parameters {
    return (
        isSimulationParameters(parameters) ||
        isOptimalControlParameters(parameters) ||
        isPIParameters(parameters)
    );
}
