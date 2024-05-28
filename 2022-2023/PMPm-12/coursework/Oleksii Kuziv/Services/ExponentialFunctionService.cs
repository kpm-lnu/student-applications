using System;
using CircuitBreakerSimulator.Models;
using CircuitBreakerSimulator.Services.Interfaces;

namespace CircuitBreakerSimulator.Services;

public class ExponentialFunctionService : IFunctionService
{
    private readonly double _coefficientA;
    private readonly double _coefficientB;
    private readonly double _errorProbability;
    private readonly CircuitBreaker _circuitBreaker;

    public ExponentialFunctionService()
    {
        _coefficientA = 1;
        _coefficientB = 1;
        _errorProbability = 0.1;
    }

    public ExponentialFunctionService(FunctionCoefficients coefficients, FunctionServiceOptions options, CircuitBreaker circuitBreaker)
    {
        _coefficientA = coefficients.CoefficientA;
        _coefficientB = coefficients.CoefficientB;
        _errorProbability = options.ErrorProbability;
        _circuitBreaker = circuitBreaker;
    }

    public double Calculate(double x)
    {
        if (!_circuitBreaker.IsRequestAllowed())
        {
            return double.NaN;
        }

        if (ShouldGenerateError())
        {
            return double.NaN;
        }

        _circuitBreaker.MarkSuccess();

        return _coefficientA * Math.Exp(_coefficientB * x);
    }

    public CircuitBreaker GetBreaker()
    {
        return _circuitBreaker;
    }

    private bool ShouldGenerateError()
    {
        var random = new Random();
        var randomValue = random.NextDouble();
        return randomValue < _errorProbability;
    }
}