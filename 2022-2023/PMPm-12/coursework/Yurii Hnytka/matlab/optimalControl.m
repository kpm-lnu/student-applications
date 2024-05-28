function data = optimalControl(modelJson)
    eps = 1e-3;
    iterations = 0;

    % ========================================================================= %

    payload = jsondecode(modelJson);

    data = struct();
    data.time = payload.simulationParameters.time;
    data.step = payload.simulationParameters.step;
    data.compartments = {};

    simulationNodesAmount = payload.simulationParameters.time / payload.simulationParameters.step;

    compartmentEquations = getCompartmentsEquations(payload);

    for i = 1:numel(payload.model)
        payload.model(i).values = [payload.model(i).value, zeros(1, simulationNodesAmount - 1)];
        payload.model(i).equation = compartmentEquations(i);
    end

    payload.costFunction = getCostFunction(payload);
    payload.hamiltonian = getHamiltonian(payload);

    interventionDerivatives = getInterventionDerivatives(payload);

    for i = 1:numel(payload.interventions)
        payload.interventions(i).values = zeros(1, simulationNodesAmount);
        payload.interventions(i).equation = interventionDerivatives(i);
    end

    lambdaDerivatives = getLambdaDerivatives(payload);

    payload.lambdas = [];

    for i = 1:numel(payload.model)
        payload.lambdas = [payload.lambdas, struct('equation', lambdaDerivatives(i), 'values', zeros(1, simulationNodesAmount))];
    end

    % ========================================================================= %

    simulationResults = sir_optimised(payload);

    for i = 1:numel(payload.model)
        payload.model(i).values = simulationResults(i).values;
    end

    prevCost = 0;
    currentCost = costFunc(payload);
    startCost = currentCost;

    while true
        previousSimulationResults = simulationResults;
        prevLambdas = payload.lambdas.values;

        newLambdas = lambdaSystem(payload);

        for i = 1:numel(payload.model)
            payload.lambdas(i).values = newLambdas(i, :);
        end

        interventionsValues = updateUFunctions(payload);

        for i = 1:numel(payload.interventions)
            payload.interventions(i).values = interventionsValues(i, :);
        end

        for i = 1:numel(payload.model)
            payload.model(i).values = [payload.model(i).value, zeros(1, simulationNodesAmount - 1)];
        end

        simulationResults = sir_optimised(payload);

        for i = 1:numel(payload.model)
            payload.model(i).values = simulationResults(i).values;
        end

        prevCost = currentCost;
        currentCost = costFunc(payload);

        iterations = iterations + 1;

        if (abs(prevCost - currentCost) <= eps)
            break;
        end
    end

    for i = 1:numel(payload.model)
        dataCompartment = struct();

        dataCompartment.id = payload.model(i).id;
        dataCompartment.values = payload.model(i).values;

        data.compartments = [data.compartments, dataCompartment];
    end

    for i = 1:numel(payload.interventions)
        dataIntervention = struct();

        dataIntervention.id = payload.interventions(i).id;
        dataIntervention.values = payload.interventions(i).values;

        data.compartments = [data.compartments, dataIntervention];
    end
end

% ========================================================================= %

function modelData = sir_optimised(payload)
    variablesData = getVariablesData(payload);

    for t = 1:payload.simulationParameters.time / payload.simulationParameters.step - 1
        for i = 1:numel(payload.model)
            compartmentId = payload.model(i).id;
            values = cell(1, numel(payload.model(i).equation.vars));

            for j = 1:numel(payload.model(i).equation.vars)
                values{j} = variablesData.(payload.model(i).equation.vars{j})(t);
            end

            variablesData.(compartmentId)(t + 1) = variablesData.(compartmentId)(t) + payload.model(i).equation.equation(values{:});
        end
    end

    modelData = [];

    for i = 1:numel(payload.model)
        compartmentValues = struct();

        compartmentValues.id = payload.model(i).id;
        compartmentValues.values = variablesData.(payload.model(i).id);

        modelData = [modelData, compartmentValues];
    end
end

function lambdas = lambdaSystem(payload)
    variablesData = getVariablesData(payload);

    for t = payload.simulationParameters.time/payload.simulationParameters.step - 1:-1:1
        for i = 1:numel(payload.model)
            lambdaId = strcat('lambda', num2str(i));
            values = cell(1, numel(payload.lambdas(i).equation.vars));

            for j = 1:numel(payload.lambdas(i).equation.vars)
                values{j} = variablesData.(payload.lambdas(i).equation.vars{j})(t + 1);
            end

            variablesData.(lambdaId)(t) = variablesData.(lambdaId)(t + 1) - payload.lambdas(i).equation.equation(values{:});
        end
    end

    lambdas = [];

    for i = 1:numel(payload.model)
        lambdas = [lambdas; variablesData.(strcat('lambda', num2str(i)))];
    end
end

function interventions = updateUFunctions(payload)
    variablesData = getVariablesData(payload);

    for t = 1:payload.simulationParameters.time/payload.simulationParameters.step
        for i = 1:numel(payload.interventions)
            interventionId = payload.interventions(i).id;
            values = cell(1, numel(payload.interventions(i).equation.vars));

            for j = 1:numel(payload.interventions(i).equation.vars)
                values{j} = variablesData.(payload.interventions(i).equation.vars{j})(t);
            end

            variablesData.(interventionId)(t) = 0.5 * ( ...
                variablesData.(interventionId)(t) + ...
                min( ...
                    0.9, ...
                    max( ...
                        0, ...
                        payload.interventions(i).equation.equation(values{:}) ...
                    ) ...
                ) ...
            );
        end
    end

    interventions = [];

    for i = 1:numel(payload.interventions)
        interventions = [interventions; variablesData.(payload.interventions(i).id)];
    end
end

function cost = costFunc(payload)
    variablesData = getVariablesData(payload);

    cost = 0;

    for t = 1:payload.simulationParameters.time/payload.simulationParameters.step
        values = cell(1, numel(payload.costFunction.vars));

        for i = 1:numel(payload.costFunction.vars)
            values{i} = variablesData.(payload.costFunction.vars{i})(t);
        end

        cost = cost + payload.costFunction.equation(values{:});
    end
end

% ========================================================================= %

function variablesData = getVariablesData(payload)
    variablesData = struct();

    for i = 1:numel(payload.model)
        variablesData.(payload.model(i).id) = payload.model(i).values;
        variablesData.(strcat('lambda', num2str(i))) = payload.lambdas(i).values;
    end

    for i = 1:numel(payload.interventions)
        variablesData.(payload.interventions(i).id) = payload.interventions(i).values;
    end
end

function hamiltonian = getHamiltonian(payload)
    hamiltonianEquation = payload.costFunction.symbolicEquation;

    for i = 1:numel(payload.model)
        hamiltonianEquation = hamiltonianEquation + sym(strcat('lambda', num2str(i))) * payload.model(i).equation.symbolicEquation;
    end

    vars = string(symvar(hamiltonianEquation));

    hamiltonian = struct( ...
        'vars', vars, ...
        'equation', matlabFunction(hamiltonianEquation, 'Vars', vars), ...
        'symbolicEquation', hamiltonianEquation ...
    );
end

function costFunction = getCostFunction(payload)
    costFunctionEquation = str2sym(payload.costFunction);

    vars = string(symvar(costFunctionEquation));

    costFunction = struct( ...
        'vars', vars, ...
        'equation', matlabFunction(costFunctionEquation, 'Vars', vars), ...
        'symbolicEquation', costFunctionEquation ...
    );
end

function equations = getCompartmentsEquations(payload)
    equations = [];

    for i = 1:numel(payload.model)
        compartmentEquation = 0;

        for j = 1:numel(payload.model(i).inflows)
            inflow = payload.model(i).inflows(j);

            compartmentEquation = compartmentEquation + str2sym(inflow);
        end

        for j = 1:numel(payload.model(i).outflows)
            outflow = payload.model(i).outflows(j);

            compartmentEquation = compartmentEquation - str2sym(outflow);
        end

        vars = string(symvar(compartmentEquation));

        equation = struct( ...
            'vars', vars, ...
            'equation', matlabFunction(compartmentEquation, 'Vars', vars), ...
            'symbolicEquation', compartmentEquation ...
        );

        equations = [equations; equation];
    end
end

function lambdaDerivatives = getLambdaDerivatives(payload)
    lambdaDerivatives = [];

    for i = 1:numel(payload.model)
        lambdaDerivative = - diff(payload.hamiltonian.symbolicEquation, payload.model(i).id);

        vars = string(symvar(lambdaDerivative));

        lambdaDerivative = struct( ...
            'vars', vars, ...
            'equation', matlabFunction(lambdaDerivative, 'Vars', vars), ...
            'symbolicEquation', lambdaDerivative ...
        );

        lambdaDerivatives = [lambdaDerivatives; lambdaDerivative];
    end
end

function interventionDerivatives = getInterventionDerivatives(payload)
    interventionDerivatives = [];

    for i = 1:numel(payload.interventions)
        interventionDerivative = solve(diff(payload.hamiltonian.symbolicEquation, payload.interventions(i).id), sym(payload.interventions(i).id));

        vars = string(symvar(interventionDerivative));

        interventionDerivative = struct( ...
            'vars', vars, ...
            'equation', matlabFunction(interventionDerivative, 'Vars', vars), ...
            'symbolicEquation', interventionDerivative ...
        );

        interventionDerivatives = [interventionDerivatives; interventionDerivative];
    end
end
