using System;
using CircuitBreakerSimulator.Models;
using CircuitBreakerSimulator.Services.Interfaces;

namespace CircuitBreakerSimulator.Services;

public class CubicFunctionService : IFunctionService
{
    private readonly double _coefficientA;
    private readonly double _coefficientB;
    private readonly double _coefficientC;
    private readonly double _coefficientD;
    private readonly double _errorProbability;
    private readonly CircuitBreaker _circuitBreaker;

    public CubicFunctionService(FunctionCoefficients coefficients, FunctionServiceOptions options, CircuitBreaker circuitBreaker)
    {
        _coefficientA = coefficients.CoefficientA;
        _coefficientB = coefficients.CoefficientB;
        _coefficientC = coefficients.CoefficientC;
        _coefficientD = coefficients.CoefficientD;
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

        return _coefficientA * Math.Pow(x, 3) + _coefficientB * Math.Pow(x, 2) + _coefficientC * x + _coefficientD;
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