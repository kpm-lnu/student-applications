using System.Diagnostics;
using System.Text.RegularExpressions;
using DynamicExpresso;
using MonteCarloWeb.Models;

namespace MonteCarloWeb.Services;

public class IntegrationService
{
    public List<IntegrationResult> Integrate(IntegrationRequest request)
    {
        string expression = PrepareExpression(request.Function, request.Dimensions);

        var interpreter = new Interpreter();
        RegisterMathFunctions(interpreter);

        var lambda = interpreter.Parse(expression, Enumerable.Range(1, request.Dimensions)
            .Select(i => new Parameter($"x{i}", typeof(double))).ToArray());

        Func<double[], double> f = (double[] x) =>
        {
            var parameters = Enumerable.Range(0, x.Length)
                .Select(i => new Parameter($"x{i + 1}", x[i]))
                .ToArray();
            return Convert.ToDouble(lambda.Invoke(parameters));
        };

        double volume = Math.Pow(request.UpperBound - request.LowerBound, request.Dimensions);
        Random rand = new();

        var randomPoints = new List<double[]>();
        int maxN = request.NArray.Max();

        for (int i = 0; i < maxN; i++)
        {
            double[] point = new double[request.Dimensions];
            for (int j = 0; j < request.Dimensions; j++)
                point[j] = rand.NextDouble() * (request.UpperBound - request.LowerBound) + request.LowerBound;
            randomPoints.Add(point);
        }

        var results = new List<IntegrationResult>();

        foreach (var n in request.NArray)
        {
            var points = randomPoints.Take(n).ToArray();
            double[] functionValues = new double[n];

            Stopwatch sw = Stopwatch.StartNew();

            Parallel.For(0, n, i =>
            {
                functionValues[i] = f(points[i]);
            });

            var valid = functionValues.Where(v => !double.IsNaN(v)).ToList();
            if (valid.Count == 0)
            {
                results.Add(new IntegrationResult
                {
                    N = n,
                    I = double.NaN,
                    Sigma = double.NaN,
                    TimeSeconds = sw.Elapsed.TotalSeconds
                });
                continue;
            }

            double avg = valid.Average();
            double I = volume * avg;

            double variance = valid.Sum(v => Math.Pow(v - avg, 2)) / (valid.Count * (valid.Count - 1));
            variance = Math.Max(variance, 0);
            double sigma = Math.Sqrt(variance);

            sw.Stop();

            results.Add(new IntegrationResult
            {
                N = n,
                I = I,
                Sigma = sigma,
                TimeSeconds = sw.Elapsed.TotalSeconds
            });
        }

        return results;
    }

    private static void RegisterMathFunctions(Interpreter interpreter)
    {
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


    }

    private static string PrepareExpression(string input, int dimensions)
    {
        input = ExpandCustomFunctions(input, dimensions);
        input = ExpandCustomFunctions(input, dimensions);
        return ConvertPowers(input);
    }

    private static string ExpandCustomFunctions(string input, int dimensions)
    {
        input = Regex.Replace(input, @"Sum\s*\(\s*xi\s*\)", m =>
            string.Join("+", Enumerable.Range(1, dimensions).Select(i => $"x{i}")));

        input = Regex.Replace(input, @"Sum\s*\(\s*xi\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", m =>
        {
            int a = int.Parse(m.Groups[1].Value);
            int b = int.Parse(m.Groups[2].Value);
            if (a > b) (a, b) = (b, a);
            return string.Join("+", Enumerable.Range(a, b - a + 1).Select(i => $"x{i}"));
        });

        input = Regex.Replace(input, @"Avg\s*\(\s*xi\s*\)", m =>
            $"({string.Join("+", Enumerable.Range(1, dimensions).Select(i => $"x{i}") )})/{dimensions}");

        input = Regex.Replace(input, @"Avg\s*\(\s*xi\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", m =>
        {
            int a = int.Parse(m.Groups[1].Value);
            int b = int.Parse(m.Groups[2].Value);
            if (a > b) (a, b) = (b, a);
            int count = b - a + 1;
            return $"({string.Join("+", Enumerable.Range(a, count).Select(i => $"x{i}") )})/{count}";
        });

        input = Regex.Replace(input, @"Prod\s*\(\s*xi\s*\)", m =>
            string.Join("*", Enumerable.Range(1, dimensions).Select(i => $"x{i}")));

        input = Regex.Replace(input, @"Prod\s*\(\s*xi\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", m =>
        {
            int a = int.Parse(m.Groups[1].Value);
            int b = int.Parse(m.Groups[2].Value);
            if (a > b) (a, b) = (b, a);
            return string.Join("*", Enumerable.Range(a, b - a + 1).Select(i => $"x{i}"));
        });

        input = Regex.Replace(input, @"Min\s*\(\s*xi\s*\)", m =>
            $"min({string.Join(",", Enumerable.Range(1, dimensions).Select(i => $"x{i}") )})");

        input = Regex.Replace(input, @"Max\s*\(\s*xi\s*\)", m =>
            $"max({string.Join(",", Enumerable.Range(1, dimensions).Select(i => $"x{i}") )})");

        input = Regex.Replace(input, @"Sign\s*\(([^()]+|(\([^()]*\)))+\)", m =>
        {
            string inner = m.Value.Substring(4).Trim().Trim('(', ')');
            return $"sign({inner})";
        });

        while (Regex.IsMatch(input, @"Mod\s*\(([^()]+|(\([^()]*\)))+\)"))
        {
            input = Regex.Replace(input, @"Mod\s*\(([^()]+|(\([^()]*\)))+\)", m =>
            {
                string inner = m.Value.Substring(4).Trim().Trim('(', ')');
                return $"abs({inner})";
            });
        }

        input = Regex.Replace(input, @"Sum\s*\(\s*xi\s*\^\s*\(\s*i\s*([\+\-]\s*\d+)?\s*\)\s*\)", m =>
        {
            string offsetStr = m.Groups[1].Value.Replace(" ", "");
            int offset = string.IsNullOrEmpty(offsetStr) ? 0 : int.Parse(offsetStr);
            return string.Join("+", Enumerable.Range(1, dimensions)
                .Select(i => $"Pow(x{i},{i + offset})"));
        });

        input = Regex.Replace(input, @"Prod\s*\(\s*xi\s*\^\s*\(\s*i\s*([\+\-]\s*\d+)?\s*\)\s*\)", m =>
        {
            string offsetStr = m.Groups[1].Value.Replace(" ", "");
            int offset = string.IsNullOrEmpty(offsetStr) ? 0 : int.Parse(offsetStr);
            return string.Join("*", Enumerable.Range(1, dimensions)
                .Select(i => $"Pow(x{i},{i + offset})"));
        });

        input = Regex.Replace(input, @"Sum\s*\(\s*(.*?)\s*\)", m =>
        {
            string expr = m.Groups[1].Value;
            return string.Join("+", Enumerable.Range(1, dimensions)
                .Select(i => expr.Replace("xi", $"x{i}").Replace("i", i.ToString())));
        });

        input = Regex.Replace(input, @"Prod\s*\(\s*(.*?)\s*\)", m =>
        {
            string expr = m.Groups[1].Value;
            return string.Join("*", Enumerable.Range(1, dimensions)
                .Select(i => expr.Replace("xi", $"x{i}").Replace("i", i.ToString())));
        });

        return input;
    }

    private static string ConvertPowers(string input)
    {
        string[] functions = {
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh",
            "log", "log10", "exp", "sqrt", "abs", "floor",
            "ceil", "round", "min", "max"
        };

        foreach (string func in functions)
        {
            input = Regex.Replace(input,
                @$"\b{func}\s*\^\s*(?<exp>\d+)\s*\(\s*(?<arg>[^\)]+)\s*\)",
                m => $"Pow({func}({m.Groups["arg"].Value}), {m.Groups["exp"].Value})");
        }

        input = Regex.Replace(
            input,
            @"(?<base>\([^\)]+\)|\w+)\s*\^\s*(?<exp>\([^\)]+\)|-?\d+|\w+)",
            m => $"Pow({m.Groups["base"].Value}, {m.Groups["exp"].Value})");

        return input;
    }
}
