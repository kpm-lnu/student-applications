function data=simulate(modelJson)
    % Parse the model to a struct
    model = jsondecode(modelJson);

    % Initialize the data struct
    data = struct();

    % Setting output init data
    data.time = model.simulationParameters.time;
    data.step = model.simulationParameters.step;
    data.compartments = {};

    % Get the amount of compartments
    compartmentsAmount = numel(model.model);

    % Initialize the compartment symbols array
    compartmentSymbols = cell(1, compartmentsAmount);

    % Initialize the compartments with their initial values and symbols
    for i = 1:compartmentsAmount
        % Fetch the compartment from the model
        modelCompartment = model.model(i);

        % Initialize output compartment
        dataCompartment = struct();

        % Initialize the compartment values
        dataCompartment.id = modelCompartment.id;
        dataCompartment.values = zeros(1, data.time/data.step);

        % Set the symbol
        compartmentSymbols{i} = sym(dataCompartment.id);

        % Set the initial value
        dataCompartment.values(1) = modelCompartment.value;

        % Add the compartment to the output
        data.compartments = [data.compartments, dataCompartment];
    end

    % Simulate the model
    for t = 1:data.time/data.step - 1
        % Initialize the compartment values array
        compartmentValues = cell(1, compartmentsAmount);

        % Set the compartment values
        for i = 1:compartmentsAmount
            compartmentValues{i} = data.compartments{i}.values(t);
        end

        % Calculate the new compartment values
        for i = 1:compartmentsAmount
            % Fetch the compartment from the model
            modelCompartment = model.model(i);

            % Get the amount of inflows and outflows
            compartmentInflows = numel(modelCompartment.inflows);
            compartmentOutflows = numel(modelCompartment.outflows);

            % Initialize the derivative
            d = 0;

            % Calculate inflow part of the derivative
            for j = 1:compartmentInflows
                % Fetch the inflow from the compartment
                inflow = modelCompartment.inflows(j);

                % Add the inflow to the derivative
                d = d + str2sym(inflow);
            end

            % Calculate outflow part of the derivative
            for j = 1:compartmentOutflows
                % Fetch the outflow from the compartment
                outflow = modelCompartment.outflows(j);

                % Subtract the outflow from the derivative
                d = d - str2sym(outflow);
            end

            % Calculate the new compartment value
            data.compartments{i}.values(t + 1) = data.compartments{i}.values(t) + eval(subs(d, compartmentSymbols, compartmentValues));
        end
    end
end
