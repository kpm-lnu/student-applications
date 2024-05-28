using System.Collections.Generic;
using System.Linq;

namespace CircuitBreakerSimulator.Services;

public class MetricsService
{
    private readonly int windowSize; // Розмір вікна для розрахунку ковзаючого середнього
    private List<double> responseTimes; // Список часів відгуку

    public MetricsService(int windowSize)
    {
        this.windowSize = windowSize;
        responseTimes = new List<double>();
    }

    // Метод для додавання нового значення часу відгуку
    public void AddResponseTime(double responseTime)
    {
        responseTimes.Add(responseTime);

        if (responseTimes.Count > windowSize)
        {
            responseTimes.RemoveAt(0);
        }
    }

    // Метод для розрахунку ковзаючого середнього
    public double CalculateMovingAverage()
    {
        if (responseTimes.Count > 0)
        {
            return responseTimes.Average();
        }
        else
        {
            return 0; // або будь-яке значення за замовчуванням, яке вам підходить
        }
    }
}