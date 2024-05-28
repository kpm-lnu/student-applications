function validationResult = validateCostFunction(payloadJson)
    payload = jsondecode(payloadJson);

    func = payload.func;
    allowedSymbols = payload.allowedSymbols;
    interventions = payload.interventions;

    validateExpressionResult = validateExpression(func, allowedSymbols);

    if ~validateExpressionResult.isValid
        validationResult = validateExpressionResult;

        return;
    end

    try
        symbolicExpression = str2sym(func);

        symbolsInExpression = symvar(symbolicExpression);

        variablesAvailability = ismember(symbolsInExpression, allowedSymbols)

        isExpressionContainOnlyAllowedDefinitions = all(variablesAvailability);

        if isExpressionContainOnlyAllowedDefinitions
            validationResult.isValid = true;
            validationResult.message = '';
        else
            validationResult.isValid = false;
            validationResult.message = 'The string contains variables that are not known to the system: ';

            unknownVariables = symbolsInExpression(~variablesAvailability);

            for i = 1:length(unknownVariables)
                validationResult.message = strcat(validationResult.message, '"',char(unknownVariables(i)), '"');

                if i ~= length(unknownVariables)
                    validationResult.message = strcat(validationResult.message, ', ');
                end
            end

        end
    catch
        validationResult.isValid = false;
        validationResult.message = 'The string is not a valid symbolic expression';
    end
end
