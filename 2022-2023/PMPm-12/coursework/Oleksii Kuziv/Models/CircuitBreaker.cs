using System.Collections.Generic;

namespace CircuitBreakerSimulator.Models;

public abstract class CircuitBreaker
{
    protected int failureCount;
    protected int totalRequests;

    public abstract bool IsRequestAllowed();
    public abstract void MarkSuccess();
    public abstract void MarkFailure();
    public abstract void AnalyzeResults();
    public abstract void PrintResults();

    public abstract double GetThreshold();

    public abstract List<double> GetThresholdHistory();
}