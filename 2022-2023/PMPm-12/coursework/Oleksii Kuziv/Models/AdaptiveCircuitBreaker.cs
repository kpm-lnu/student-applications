using System;
using System.Collections.Generic;
using CircuitBreakerSimulator.Services;

namespace CircuitBreakerSimulator.Models;

public class AdaptiveCircuitBreaker : CircuitBreaker
{
    private readonly int windowSize;
    private MetricsService metricsService;
    private double movingAverageThreshold;
    private TimeSpan breakDuration;
    private List<double> thresholdHistory;

    public AdaptiveCircuitBreaker(int windowSize)
    {
        this.windowSize = windowSize;
        metricsService = new MetricsService(windowSize);
        movingAverageThreshold = 0;
        breakDuration = TimeSpan.FromMinutes(1);
        thresholdHistory = new List<double>();
    }

    public override bool IsRequestAllowed()
    {
        return metricsService.CalculateMovingAverage() <= movingAverageThreshold;
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
        double averageResponseTime = metricsService.CalculateMovingAverage();
        if (averageResponseTime > movingAverageThreshold)
        {
            movingAverageThreshold *= 1.5;
            breakDuration = TimeSpan.FromMinutes(5);
        }

        thresholdHistory.Add(GetThreshold());
    }

    public override void PrintResults()
    {
        Console.WriteLine($"Failure Count: {failureCount}");
        Console.WriteLine($"Total Requests: {totalRequests}");
        Console.WriteLine($"Error Rate: {(double)failureCount / totalRequests}");
        Console.WriteLine($"Moving Average Threshold: {movingAverageThreshold}");
        Console.WriteLine($"Break Duration: {breakDuration}");
    }

    public override double GetThreshold()
    {
        return movingAverageThreshold;
    }

    public override List<double> GetThresholdHistory()
    {
        return thresholdHistory;
    }
}