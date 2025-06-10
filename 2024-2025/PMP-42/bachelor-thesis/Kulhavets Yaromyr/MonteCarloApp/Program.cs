
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using DynamicExpresso;
using ExprParameter = DynamicExpresso.Parameter;

class ClassCombined
{

     static string ExpandCustomFunctions(string input, int dimensions)
    {
        // Sum(xi)
        input = Regex.Replace(input, @"Sum\s*\(\s*xi\s*\)",
            m => string.Join("+", Enumerable.Range(1, dimensions).Select(i => $"x{i}")));

        // Sum(xi, a, b)
        input = Regex.Replace(input, @"Sum\s*\(\s*xi\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", m =>
        {
            int a = int.Parse(m.Groups[1].Value);
            int b = int.Parse(m.Groups[2].Value);
            if (a > b) (a, b) = (b, a);
            return string.Join("+", Enumerable.Range(a, b - a + 1).Select(i => $"x{i}"));
        });

        // Avg(xi)
        input = Regex.Replace(input, @"Avg\s*\(\s*xi\s*\)",
            m => $"({string.Join("+", Enumerable.Range(1, dimensions).Select(i => $"x{i}"))})/{dimensions}");

        // Avg(xi, a, b)
        input = Regex.Replace(input, @"Avg\s*\(\s*xi\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", m =>
        {
            int a = int.Parse(m.Groups[1].Value);
            int b = int.Parse(m.Groups[2].Value);
            if (a > b) (a, b) = (b, a);
            int count = b - a + 1;
            return $"({string.Join("+", Enumerable.Range(a, count).Select(i => $"x{i}"))})/{count}";
        });

        // Prod(xi)
        input = Regex.Replace(input, @"Prod\s*\(\s*xi\s*\)",
            m => string.Join("*", Enumerable.Range(1, dimensions).Select(i => $"x{i}")));

        // Prod(xi, a, b)
        input = Regex.Replace(input, @"Prod\s*\(\s*xi\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", m =>
        {
            int a = int.Parse(m.Groups[1].Value);
            int b = int.Parse(m.Groups[2].Value);
            if (a > b) (a, b) = (b, a);
            return string.Join("*", Enumerable.Range(a, b - a + 1).Select(i => $"x{i}"));
        });

        // Min(xi)
        input = Regex.Replace(input, @"Min\s*\(\s*xi\s*\)",
            m => $"min({string.Join(",", Enumerable.Range(1, dimensions).Select(i => $"x{i}"))})");

        // Max(xi)
        input = Regex.Replace(input, @"Max\s*\(\s*xi\s*\)",
            m => $"max({string.Join(",", Enumerable.Range(1, dimensions).Select(i => $"x{i}"))})");

        // Sign(expr)
        input = Regex.Replace(input, @"Sign\s*\(([^()]+|(\([^()]*\)))+\)", m =>
        {
            string inner = m.Value.Substring(4).Trim(); // remove 'Sign'
            inner = inner.Substring(1, inner.Length - 2); // remove surrounding parentheses
            return $"sign({inner})";
        });

        // Mod(expr) — рекурсивно
        while (Regex.IsMatch(input, @"Mod\s*\(([^()]+|(\([^()]*\)))+\)"))
        {
            input = Regex.Replace(input, @"Mod\s*\(([^()]+|(\([^()]*\)))+\)", m =>
            {
                string inner = m.Value.Substring(4).Trim(); // remove 'Mod'
                inner = inner.Substring(1, inner.Length - 2); // remove surrounding parentheses
                return $"abs({inner})";
            });
        }

        // Sum(xi^(i+p))
        input = Regex.Replace(input, @"Sum\s*\(\s*xi\s*\^\s*\(\s*i\s*([\+\-]\s*\d+)?\s*\)\s*\)", m =>
        {
            string offsetStr = m.Groups[1].Value.Replace(" ", "");
            int offset = 0;
            if (!string.IsNullOrEmpty(offsetStr))
                offset = int.Parse(offsetStr);

            return string.Join("+", Enumerable.Range(1, dimensions)
                .Select(i => $"Pow(x{i},{i + offset})"));
        });

        // Prod(xi^(i+p))
        input = Regex.Replace(input, @"Prod\s*\(\s*xi\s*\^\s*\(\s*i\s*([\+\-]\s*\d+)?\s*\)\s*\)", m =>
        {
            string offsetStr = m.Groups[1].Value.Replace(" ", "");
            int offset = 0;
            if (!string.IsNullOrEmpty(offsetStr))
                offset = int.Parse(offsetStr);

            return string.Join("*", Enumerable.Range(1, dimensions)
                .Select(i => $"Pow(x{i},{i + offset})"));
        });

        // Sum(expr(i)) 
        input = Regex.Replace(input, @"Sum\s*\(\s*(.*?)\s*\)", m =>
        {
            string expr = m.Groups[1].Value;
            return string.Join("+", Enumerable.Range(1, dimensions)
                .Select(i => expr.Replace("xi", $"x{i}").Replace("i", i.ToString())));
        });

        // Prod(expr(i))
        input = Regex.Replace(input, @"Prod\s*\(\s*(.*?)\s*\)", m =>
        {
            string expr = m.Groups[1].Value;
            return string.Join("*", Enumerable.Range(1, dimensions)
                .Select(i => expr.Replace("xi", $"x{i}").Replace("i", i.ToString())));
        });


        return input;
    }


    static void Main(string[] args)
    {
        Console.WriteLine("Введiть через пробiл кiлькiсть випадкових точок (наприклад: 100 200 300):");
        string input = Console.ReadLine();

        if (string.IsNullOrWhiteSpace(input))
        {
            Console.WriteLine("Неправильне введення.");
            return;
        }

        string[] nValues = input.Split(' ');
        int[] nArray;
        try
        {
            nArray = nValues.Select(int.Parse).ToArray();
        }
        catch
        {
            Console.WriteLine("Неправильне введення.");
            return;
        }

        Console.Write("Нижня межа iнтегрування: ");
        if (!double.TryParse(Console.ReadLine(), NumberStyles.Float, CultureInfo.InvariantCulture, out double c))
        {
            Console.WriteLine("Неправильне введення.");
            return;
        }

        Console.Write("Верхня межа iнтегрування: ");
        if (!double.TryParse(Console.ReadLine(), NumberStyles.Float, CultureInfo.InvariantCulture, out double d))
        {
            Console.WriteLine("Неправильне введення.");
            return;
        }

        Console.Write("Кiлькiсть змiнних (розмiрнiсть): ");
        if (!int.TryParse(Console.ReadLine(), out int dimensions))
        {
            Console.WriteLine("Неправильне введення.");
            return;
        }

        static string ConvertPowers(string input)
        {
            string[] functions = {
                "sin", "cos", "tan", "asin", "acos", "atan",
                "sinh", "cosh", "tanh",
                "log", "log10", "exp", "sqrt", "abs", "floor",
                "ceil", "round", "min", "max"
            };

            foreach (string func in functions)
            {
                input = Regex.Replace(
                    input,
                    @$"\b{func}\s*\^\s*(?<exp>\d+)\s*\(\s*(?<arg>[^\)]+)\s*\)",
                    m => $"Pow({func}({m.Groups["arg"].Value}), {m.Groups["exp"].Value})");
            }

            input = Regex.Replace(
                input,
                @"(?<base>\([^\)]+\)|\w+)\s*\^\s*(?<exp>\([^\)]+\)|-?\d+|\w+)",
                m => $"Pow({m.Groups["base"].Value}, {m.Groups["exp"].Value})");

            return input;
        }

        Console.WriteLine("Введiть формулу функцiї f(x1, x2, ..., xn):");
        string functionExpression = Console.ReadLine();
        functionExpression = ExpandCustomFunctions(functionExpression, dimensions);
        functionExpression = ExpandCustomFunctions(functionExpression, dimensions);
        functionExpression = ConvertPowers(functionExpression);

        var interpreter = new Interpreter();
        interpreter.SetFunction("Pow", new Func<double, double, double>(Math.Pow));
        interpreter.SetFunction("log", new Func<double, double>(Math.Log));
        interpreter.SetFunction("log10", new Func<double, double>(Math.Log10));
        interpreter.SetFunction("exp", new Func<double, double>(Math.Exp));
        interpreter.SetFunction("sqrt", new Func<double, double>(Math.Sqrt));
        interpreter.SetFunction("abs", new Func<double, double>(Math.Abs));
        interpreter.SetFunction("floor", new Func<double, double>(Math.Floor));
        interpreter.SetFunction("ceil", new Func<double, double>(Math.Ceiling));
        interpreter.SetFunction("round", new Func<double, double>(Math.Round));
        interpreter.SetFunction("min", new Func<double, double, double>(Math.Min));
        interpreter.SetFunction("max", new Func<double, double, double>(Math.Max));
        interpreter.SetFunction("sin", new Func<double, double>(Math.Sin));
        interpreter.SetFunction("cos", new Func<double, double>(Math.Cos));
        interpreter.SetFunction("tan", new Func<double, double>(Math.Tan));
        interpreter.SetFunction("asin", new Func<double, double>(Math.Asin));
        interpreter.SetFunction("acos", new Func<double, double>(Math.Acos));
        interpreter.SetFunction("atan", new Func<double, double>(Math.Atan));
        interpreter.SetFunction("sinh", new Func<double, double>(Math.Sinh));
        interpreter.SetFunction("cosh", new Func<double, double>(Math.Cosh));
        interpreter.SetFunction("tanh", new Func<double, double>(Math.Tanh));

          var lambda = interpreter.Parse(functionExpression, Enumerable.Range(1, dimensions)
         .Select(i => new ExprParameter($"x{i}", typeof(double))).ToArray());

            Func<double[], double> f = (double[] x) =>
            {
                var parameters = Enumerable.Range(0, x.Length)
                    .Select(i => new ExprParameter($"x{i + 1}", x[i]))
                    .ToArray();

                return Convert.ToDouble(lambda.Invoke(parameters));
            };


        double volume = Math.Pow(d - c, dimensions);

        Random rand = new Random();
        List<double[]> randomPoints = new List<double[]>();
        int maxN = nArray.Max();

        for (int i = 0; i < maxN; i++)
        {
            double[] point = new double[dimensions];
            for (int j = 0; j < dimensions; j++)
                point[j] = rand.NextDouble() * (d - c) + c;
            randomPoints.Add(point);
        }

        List<double> resultsI = new List<double>();
        List<double> resultsSigma = new List<double>();

        foreach (var n in nArray)
        {
            var points = randomPoints.GetRange(0, n);
            double[] functionValues = new double[n];

            Stopwatch stopwatch = Stopwatch.StartNew();

            Parallel.ForEach(Enumerable.Range(0, n), i =>
            {
                functionValues[i] = f(points[i]);
            });

            var validValues = functionValues.Where(v => !double.IsNaN(v)).ToList();
            if (validValues.Count == 0)
            {
                Console.WriteLine("Усi значення функції стали NaN. Можливо, область інтегрування містить недопустимі значення.");
                resultsI.Add(double.NaN);
                resultsSigma.Add(double.NaN);
                continue;
            }

            double sum = validValues.Sum();
            double I = volume * (sum / validValues.Count);

            double diffSquareSum = validValues.Sum(val => Math.Pow(val - sum / validValues.Count, 2));
            double variance = diffSquareSum / (validValues.Count * (validValues.Count - 1));
            if (variance < 0) variance = 0;
            double sigma = Math.Sqrt(variance);

            stopwatch.Stop();

            resultsI.Add(I);
            resultsSigma.Add(sigma);
            Console.WriteLine($"n = {n}");
            Console.WriteLine($"I = {I}");
            Console.WriteLine($"Похибка Sigma = {sigma}");
            Console.WriteLine($"Час виконання: {stopwatch.Elapsed.TotalSeconds:F6} с\n");
        }

        Console.WriteLine("\nРезультати:");
        Console.WriteLine("n\tI (наближено)\t\tПохибка Sigma");
        for (int i = 0; i < nArray.Length; i++)
        {
            Console.WriteLine($"{nArray[i]}\t{resultsI[i]:F10}\t{resultsSigma[i]:F10}");
        }

        Console.WriteLine("\nНатиснiть Enter, щоб завершити програму...");
        while (Console.ReadKey(true).Key != ConsoleKey.Enter) { }
    }
}
