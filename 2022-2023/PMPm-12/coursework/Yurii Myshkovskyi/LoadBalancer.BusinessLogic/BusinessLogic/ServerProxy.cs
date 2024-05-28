public enum DelayComplexityFunction
{
    Periodic_SinCos = 0,
    Periodic_Cos2 = 1,
    Periodic_Sin = 2,
    Periodic_Sin_Plus_Cos = 3,
    Sqrt = 4,
    Linear = 5,
    Polynomial = 6,
    Exponential = 7,
}

public class ServerProxy
{
    private readonly List<long> _responseTimes = new List<long>();
    private readonly List<bool> _responseStatuses = new List<bool>();
    private readonly int _maxResponseTimeHistory = 20;
    private readonly object _syncLock = new object();
    private int _requestCounter = 0;

    public ServerProxy(DelayComplexityFunction delayComplexity)
    {
        DelayComplexity = delayComplexity;
    }

    public DelayComplexityFunction DelayComplexity { get; }

    public async Task<Response> AcceptRequestAsync(Request request)
    {
        int delayThreshold = 110;
        DateTime startTime = DateTime.UtcNow;
        var delay = GetDelayMiliseconds();
        Interlocked.Increment(ref _requestCounter);
        if (delay >= delayThreshold)
        {
            lock (_syncLock)
            {
                var calculatedTicks = (startTime.AddMilliseconds(delay) - startTime).Ticks;
                _responseTimes.Add(calculatedTicks);
                if (_responseTimes.Count > _maxResponseTimeHistory)
                {
                    _responseTimes.RemoveAt(0);
                }
                _responseStatuses.Add(false);
                if (_responseStatuses.Count > _maxResponseTimeHistory)
                {
                    _responseStatuses.RemoveAt(0);
                }
            }
            throw new Exception("Simulated server error due to high response time.");
        }
        else
        {
            await Task.Delay(delay);
            DateTime endTime = DateTime.UtcNow;
            lock (_syncLock)
            {
                _responseTimes.Add((endTime - startTime).Ticks);
                if (_responseTimes.Count > _maxResponseTimeHistory)
                {
                    _responseTimes.RemoveAt(0);
                }

                _responseStatuses.Add(true);
                if (_responseStatuses.Count > _maxResponseTimeHistory)
                {
                    _responseStatuses.RemoveAt(0);
                }
            }
            return new Response { Data = "Processed: " + request.Data };
        }
    }


    public double GetActualWeight()
    {
        lock (_syncLock)
        {
            if (_responseTimes.Count == 0)
            {
                return double.MaxValue;
            }
            var averageResponseTime = CalculateExponentialMovingAverage(_responseTimes);
            var emaWeight = (1.0 / averageResponseTime) * 1000000;
            double successRate = _requestCounter < 5 ? 1 : (double)(_responseStatuses.Count(s => s)) / (_responseStatuses.Count);
            var weight = emaWeight * Math.Pow(successRate, 3) / Math.Pow(_requestCounter, 0.5);

            return emaWeight;
        }
    }

    public double GetConstantWeight()
    {
        var complexity = Math.Pow((int)DelayComplexity + 1, 2);
        return 1.0 / (complexity);
    }

    private int GetDelayMiliseconds()
    {
        int periodicDegreeStep = 5;
        switch (DelayComplexity)
        {
            case DelayComplexityFunction.Sqrt:
                {
                    return 2 * (int)Math.Sqrt(_requestCounter);
                }
            case DelayComplexityFunction.Linear:
                {
                    return 2 * _requestCounter;
                };
            case DelayComplexityFunction.Exponential:
                {
                    if (_requestCounter < 10)
                    {
                        return 1 * (int)Math.Pow(2, _requestCounter);
                    }
                    else
                    {
                        return 1100;
                    }
                }
            case DelayComplexityFunction.Polynomial:
                {
                    if (_requestCounter < 10)
                    {
                        return 1 * (int)Math.Pow(_requestCounter, 2);
                    }
                    else
                    {
                        return 110;
                    }
                }
            case DelayComplexityFunction.Periodic_Sin_Plus_Cos:
                {
                    return (int)Math.Abs((80 * (Math.Sin(_requestCounter * periodicDegreeStep) + Math.Cos(_requestCounter * periodicDegreeStep))));
                }
            case DelayComplexityFunction.Periodic_SinCos:
                {
                    return (int)Math.Abs((230 * (Math.Sin(_requestCounter * periodicDegreeStep) * Math.Cos(_requestCounter * periodicDegreeStep))));
                }
            case DelayComplexityFunction.Periodic_Sin:
                {
                    return (int)Math.Abs(120 * Math.Sin(_requestCounter * periodicDegreeStep));
                }
            case DelayComplexityFunction.Periodic_Cos2:
                {
                    return (int)Math.Abs(120 * Math.Pow(Math.Cos(_requestCounter * periodicDegreeStep), 2));
                }

        }
        return 1000;
    }

    private double CalculateExponentialMovingAverage(List<long> responseTimes)
    {
        double smoothingFactor = 0.5; // You can adjust the smoothing factor between 0 and 1 as needed
        double ema = responseTimes[0];

        for (int i = 1; i < responseTimes.Count; i++)
        {
            ema = (responseTimes[i] * smoothingFactor) + (ema * (1 - smoothingFactor));
        }

        return ema;
    }
}