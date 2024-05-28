using System;
using CircuitBreakerSimulator.Models;
using CircuitBreakerSimulator.Services.Interfaces;

namespace CircuitBreakerSimulator.Services;

public class QuadraticFunctionService : IFunctionService
{
    private readonly double _coefficientA;
    private readonly double _coefficientB;
    private readonly double _coefficientC;
    private readonly double _errorProbability;
    private readonly CircuitBreaker _circuitBreaker;

    public QuadraticFunctionService()
    {
        _coefficientA = 1;
        _coefficientB = 1;
        _coefficientC = 1;
        _errorProbability = 0.1;
    }

    public QuadraticFunctionService(FunctionCoefficients coefficients, FunctionServiceOptions options, CircuitBreaker circuitBreaker)
    {
        _coefficientA = coefficients.CoefficientA;
        _coefficientB = coefficients.CoefficientB;
        _coefficientC = coefficients.CoefficientC;
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

        return _coefficientA * x * x + _coefficientB * x + _coefficientC;
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