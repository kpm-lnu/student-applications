using System;
using System.Collections.Generic;
using CircuitBreakerSimulator.Services;

namespace CircuitBreakerSimulator.Models;

public class StaticCircuitBreaker : CircuitBreaker
{
    private readonly double errorThreshold;
    private TimeSpan breakDuration;
    private List<double> thresholdHistory;

    public StaticCircuitBreaker(int errorThreshold, TimeSpan breakDuration)
    {
        this.errorThreshold = errorThreshold;
        this.breakDuration = breakDuration;
        thresholdHistory = new List<double>();
    }

    public override bool IsRequestAllowed()
    {
        return failureCount < errorThreshold;
    }

    public override void MarkSuccess()
    {
        totalRequests++;
    }

    public override void MarkFailure()
    {
        failureCount++;
        totalRequests++;
    }

    public override void AnalyzeResults()
    {
        double errorRate = (double)failureCount / totalRequests;
        if (errorRate > errorThreshold)
        {
            breakDuration = TimeSpan.FromMinutes(5);
        }

        thresholdHistory.Add(GetThreshold());
    }

    public override void PrintResults()
    {
        Console.WriteLine($"Failure Count: {failureCount}");
        Console.WriteLine($"Total Requests: {totalRequests}");
        Console.WriteLine($"Error Rate: {(double)failureCount / totalRequests}");
        Console.WriteLine($"Error Threshold: {errorThreshold}");
        Console.WriteLine($"Break Duration: {breakDuration}");
    }

    public override double GetThreshold()
    {
        return errorThreshold;
    }

    public override List<double> GetThresholdHistory()
    {
        return thresholdHistory;
    }
}
