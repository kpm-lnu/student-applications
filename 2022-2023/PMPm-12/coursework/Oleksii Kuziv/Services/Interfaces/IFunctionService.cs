using CircuitBreakerSimulator.Models;

namespace CircuitBreakerSimulator.Services.Interfaces;

public interface IFunctionService
{
    CircuitBreaker GetBreaker();

    double Calculate(double x);
}