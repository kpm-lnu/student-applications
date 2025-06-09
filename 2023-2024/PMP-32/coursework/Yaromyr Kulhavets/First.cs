using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;

class First
{
    static void Main(string[] args)
    {
        Console.WriteLine("Введiть через пробiл кiлькiсть випадкових чисел (наприклад: 100 200 300):");
        string input = Console.ReadLine();

        if (string.IsNullOrWhiteSpace(input))
        {
            Console.WriteLine("Неправильне введення. Перезапустiть програму та введiть дiйснi числа.");
            return;
        }

        string[] nValues = input.Split(' ');
        int[] nArray;

        try
        {
            nArray = nValues.Select(int.Parse).ToArray();
        }
        catch (FormatException)
        {
            Console.WriteLine("Неправильне введення. Перезапустiть програму та введiть дiйснi числа.");
            return;
        }

        Console.Write("Нижня межа iнтегрування: ");
        if (!double.TryParse(Console.ReadLine(), NumberStyles.Float, CultureInfo.InvariantCulture, out double c))
        {
            Console.WriteLine("Неправильне введення. Перезапустiть програму та введiть дiйснi числа.");
            return;
        }

        Console.Write("Верхня межа iнтегрування: ");
        if (!double.TryParse(Console.ReadLine(), NumberStyles.Float, CultureInfo.InvariantCulture, out double d))
        {
            Console.WriteLine("Неправильне введення. Перезапустiть програму та введiть дiйснi числа.");
            return;
        }

        // Фіксовані пари
        List<Tuple<double, double>> fixedPairs = new List<Tuple<double, double>>
        {
            Tuple.Create(0.5, 0.75),
            Tuple.Create(0.25, 0.5)
        };

        // Генеруємо пари рандомних чисел для найбільшого значення n
        Random rand = new Random();
        List<Tuple<double, double>> randomPairs = new List<Tuple<double, double>>();
        int maxN = nArray.Max(); // Знаходимо максимальне значення n

        // Генеруємо випадкові пари чисел для найбільшого значення n
        for (int i = 0; i < maxN - 2; i++)
        {
            double x = rand.NextDouble() * (d - c) + c;
            double y = rand.NextDouble() * (d - c) + c;
            randomPairs.Add(Tuple.Create(x, y));
        }

        List<double> resultsI = new List<double>();
        List<double> resultsA = new List<double>();

        foreach (int n in nArray)
        {
            List<Tuple<double, double>> pairs = new List<Tuple<double, double>>(fixedPairs);
            pairs.AddRange(randomPairs.GetRange(0, n - 2));

            Stopwatch stopwatch = Stopwatch.StartNew(); // Початок вимірювання часу

            // Обчислюємо суму квадратів x_i та y_i для кожної пари
            double sumOfSquares = 0;
            foreach (var pair in pairs)
            {
                double x = pair.Item1;
                double y = pair.Item2;
                sumOfSquares += x * x + y * y;
            }

            // Обчислюємо значення виразу I для поточного значення n
            double I = Math.Pow(d - c, 2) * (sumOfSquares / n);

            // Обчислюємо значення a для поточного значення n
            double sumDifferenceSquares = 0;
            foreach (var pair in pairs)
            {
                double x = pair.Item1;
                double y = pair.Item2;
                sumDifferenceSquares += Math.Pow(x * x + y * y - I, 2);
            }
            double variance = sumDifferenceSquares / (n * (n - 1));
            double a = variance >= 0 ? Math.Sqrt(variance) : double.NaN;

            stopwatch.Stop(); // Зупинка вимірювання часу

            resultsI.Add(I);
            resultsA.Add(a);

            Console.WriteLine($"Кiлькiсть випадкових чисел: {n}");
            Console.WriteLine($"Значення наближення iнтегралу I: {I}");
            Console.WriteLine($"Значення оцiнки похибки a: {a}");
            Console.WriteLine($"Час виконання для {n}: {stopwatch.Elapsed.TotalSeconds:F10} секунд\n");
        }

        // Записуємо результати у файл result.csv
        string fileName = "result.csv";
        using (StreamWriter writer = new StreamWriter(fileName))
        {
            writer.WriteLine("n, I, a");
            for (int i = 0; i < nArray.Length; i++)
            {
                writer.WriteLine($"{nArray[i]}, {resultsI[i]}, {resultsA[i]}");
            }
        }
        Console.WriteLine($"Результати було записано у файл {fileName}");
        Console.WriteLine($"Файл знаходиться за шляхом: {Path.GetFullPath(fileName)}");
    }
}
