using System;
using CircuitBreakerSimulator.Services.Interfaces;

namespace CircuitBreakerSimulator.Services;

public class FunctionSimulator
{
    private readonly IFunctionService _functionService;

    public FunctionSimulator(IFunctionService functionService)
    {
        _functionService = functionService;
    }

    public double[] SimulateFunctionCalls(double[] inputs)
    {
        double[] results = new double[inputs.Length];

        for (int i = 0; i < inputs.Length; i++)
        {
            try
            {
                double result = _functionService.Calculate(inputs[i]);
                results[i] = result;
            }
            catch (Exception)
            {
                results[i] = double.NaN; // Set NaN for failed calls
            }
        }

        return results;
    }
}