using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SylvesterSolver.FSharp;

Console.OutputEncoding = Encoding.UTF8;
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

var outputDir = Path.Combine(AppContext.BaseDirectory, "results");
Directory.CreateDirectory(outputDir);

Console.WriteLine("=== Extended ADI experiments ===");
Console.WriteLine($"Results directory: {outputDir}");
Console.WriteLine();

RunDimensionExperiment(outputDir);
RunToleranceExperiment(outputDir);
RunShiftExperiment(outputDir);
RunDirectComparisonExperiment(outputDir);

Console.WriteLine();
Console.WriteLine("All experiments finished.");

static void RunDimensionExperiment(string outputDir)
{
    Console.WriteLine("[1/4] Dimension impact experiment...");

    var sizes = new[] { 20, 40, 80, 120, 160, 200 };
    var runs = 7;
    var tolerance = 1e-8;
    var maxIterations = 1000;

    var csv = new StringBuilder();
    csv.AppendLine("n,runs,iter_mean,iter_std,residual_mean,residual_std,time_ms_mean,time_ms_std,converged_rate");

    foreach (var n in sizes)
    {
        var iter = new double[runs];
        var residual = new double[runs];
        var timeMs = new double[runs];
        var converged = 0;

        for (var r = 0; r < runs; r++)
        {
            var (a, b, c, xExact) = GenerateTestProblem(n, seed: 1000 + n * 10 + r);
            var pShifts = GenerateLogShiftsFromBounds(a, n);
            var qShifts = GenerateLogShiftsFromBounds(b, n);

            if (r == 0)
            {
                _ = SylvesterAdiSolver.Solve(a, b, c, pShifts, qShifts, tolerance, maxIterations);
            }

            var sw = Stopwatch.StartNew();
            var result = SylvesterAdiSolver.Solve(a, b, c, pShifts, qShifts, tolerance, maxIterations);
            sw.Stop();

            iter[r] = result.Iterations;
            residual[r] = result.ResidualNorm;
            timeMs[r] = sw.Elapsed.TotalMilliseconds;
            if (result.Converged) converged++;

            _ = ComputeError(xExact, result.Solution, n);
        }

        csv.AppendLine(string.Join(",",
            n,
            runs,
            Mean(iter).ToString("F2"),
            Std(iter).ToString("F2"),
            Mean(residual).ToString("E3"),
            Std(residual).ToString("E3"),
            Mean(timeMs).ToString("F2"),
            Std(timeMs).ToString("F2"),
            ((double)converged / runs).ToString("F2")));

        Console.WriteLine($"  n={n}: iter={Mean(iter):F1}±{Std(iter):F1}, time={Mean(timeMs):F1}±{Std(timeMs):F1} ms");
    }

    File.WriteAllText(Path.Combine(outputDir, "dimension_effect.csv"), csv.ToString());
}

static void RunToleranceExperiment(string outputDir)
{
    Console.WriteLine("[2/4] Tolerance impact experiment...");

    var n = 120;
    var tolerances = new[] { 1e-6, 1e-8, 1e-10, 1e-12 };
    var runs = 7;
    var maxIterations = 1500;

    var csv = new StringBuilder();
    csv.AppendLine("n,tolerance,runs,iter_mean,iter_std,residual_mean,residual_std,error_mean,error_std,time_ms_mean,time_ms_std,converged_rate");

    foreach (var tol in tolerances)
    {
        var iter = new double[runs];
        var residual = new double[runs];
        var errors = new double[runs];
        var timeMs = new double[runs];
        var converged = 0;

        for (var r = 0; r < runs; r++)
        {
            var (a, b, c, xExact) = GenerateTestProblem(n, seed: 2000 + r);
            var pShifts = GenerateLogShiftsFromBounds(a, n);
            var qShifts = GenerateLogShiftsFromBounds(b, n);

            if (r == 0)
            {
                _ = SylvesterAdiSolver.Solve(a, b, c, pShifts, qShifts, tol, maxIterations);
            }

            var sw = Stopwatch.StartNew();
            var result = SylvesterAdiSolver.Solve(a, b, c, pShifts, qShifts, tol, maxIterations);
            sw.Stop();

            iter[r] = result.Iterations;
            residual[r] = result.ResidualNorm;
            errors[r] = ComputeError(xExact, result.Solution, n);
            timeMs[r] = sw.Elapsed.TotalMilliseconds;
            if (result.Converged) converged++;
        }

        csv.AppendLine(string.Join(",",
            n,
            tol.ToString("E0"),
            runs,
            Mean(iter).ToString("F2"),
            Std(iter).ToString("F2"),
            Mean(residual).ToString("E3"),
            Std(residual).ToString("E3"),
            Mean(errors).ToString("E3"),
            Std(errors).ToString("E3"),
            Mean(timeMs).ToString("F2"),
            Std(timeMs).ToString("F2"),
            ((double)converged / runs).ToString("F2")));

        Console.WriteLine($"  tol={tol:E0}: iter={Mean(iter):F1}±{Std(iter):F1}, residual={Mean(residual):E2}");
    }

    File.WriteAllText(Path.Combine(outputDir, "tolerance_effect.csv"), csv.ToString());
}

static void RunShiftExperiment(string outputDir)
{
    Console.WriteLine("[3/4] Shift strategy impact experiment...");

    var n = 120;
    var runs = 7;
    var tolerance = 1e-8;
    var maxIterations = 1500;
    var strategies = new[] { "uniform", "log", "spectral" };

    var csv = new StringBuilder();
    csv.AppendLine("n,strategy,runs,iter_mean,iter_std,residual_mean,residual_std,time_ms_mean,time_ms_std,error_mean,error_std,converged_rate");

    foreach (var strategy in strategies)
    {
        var iter = new double[runs];
        var residual = new double[runs];
        var errors = new double[runs];
        var timeMs = new double[runs];
        var converged = 0;

        for (var r = 0; r < runs; r++)
        {
            var (a, b, c, xExact) = GenerateTestProblem(n, seed: 3000 + r);
            var (pShifts, qShifts) = strategy switch
            {
                "uniform" => (GenerateUniformShifts(n, scale: 1.0), GenerateUniformShifts(n, scale: 0.8)),
                "log" => (GenerateLogShiftsFromBounds(a, n), GenerateLogShiftsFromBounds(b, n)),
                _ => (GenerateSpectralAdaptiveShifts(a, n), GenerateSpectralAdaptiveShifts(b, n))
            };

            if (r == 0)
            {
                _ = SylvesterAdiSolver.Solve(a, b, c, pShifts, qShifts, tolerance, maxIterations);
            }

            var sw = Stopwatch.StartNew();
            var result = SylvesterAdiSolver.Solve(a, b, c, pShifts, qShifts, tolerance, maxIterations);
            sw.Stop();

            iter[r] = result.Iterations;
            residual[r] = result.ResidualNorm;
            errors[r] = ComputeError(xExact, result.Solution, n);
            timeMs[r] = sw.Elapsed.TotalMilliseconds;
            if (result.Converged) converged++;
        }

        csv.AppendLine(string.Join(",",
            n,
            strategy,
            runs,
            Mean(iter).ToString("F2"),
            Std(iter).ToString("F2"),
            Mean(residual).ToString("E3"),
            Std(residual).ToString("E3"),
            Mean(timeMs).ToString("F2"),
            Std(timeMs).ToString("F2"),
            Mean(errors).ToString("E3"),
            Std(errors).ToString("E3"),
            ((double)converged / runs).ToString("F2")));

        Console.WriteLine($"  {strategy}: iter={Mean(iter):F1}±{Std(iter):F1}, time={Mean(timeMs):F1}±{Std(timeMs):F1} ms");
    }

    File.WriteAllText(Path.Combine(outputDir, "shift_effect.csv"), csv.ToString());
}

static void RunDirectComparisonExperiment(string outputDir)
{
    Console.WriteLine("[4/4] ADI vs direct method (small n)...");

    var sizes = new[] { 4, 6, 8, 10, 12 };
    var runs = 5;
    var tolerance = 1e-10;
    var maxIterations = 1200;

    var csv = new StringBuilder();
    csv.AppendLine("n,runs,adi_iter_mean,adi_iter_std,adi_time_ms_mean,adi_time_ms_std,direct_time_ms_mean,direct_time_ms_std,adi_error_to_exact_mean,direct_error_to_exact_mean,adi_direct_diff_mean");

    foreach (var n in sizes)
    {
        var adiIter = new double[runs];
        var adiTime = new double[runs];
        var directTime = new double[runs];
        var adiErr = new double[runs];
        var directErr = new double[runs];
        var crossErr = new double[runs];

        for (var r = 0; r < runs; r++)
        {
            var (a, b, c, xExact) = GenerateTestProblem(n, seed: 4000 + n * 10 + r);
            var pShifts = GenerateLogShiftsFromBounds(a, n);
            var qShifts = GenerateLogShiftsFromBounds(b, n);

            if (r == 0)
            {
                _ = SylvesterAdiSolver.Solve(a, b, c, pShifts, qShifts, tolerance, maxIterations);
                _ = SolveSylvesterDirect(a, b, c);
            }

            var swAdi = Stopwatch.StartNew();
            var adi = SylvesterAdiSolver.Solve(a, b, c, pShifts, qShifts, tolerance, maxIterations);
            swAdi.Stop();

            var swDir = Stopwatch.StartNew();
            var xDirect = SolveSylvesterDirect(a, b, c);
            swDir.Stop();

            adiIter[r] = adi.Iterations;
            adiTime[r] = swAdi.Elapsed.TotalMilliseconds;
            directTime[r] = swDir.Elapsed.TotalMilliseconds;
            adiErr[r] = ComputeError(xExact, adi.Solution, n);
            directErr[r] = ComputeError(xExact, xDirect, n);
            crossErr[r] = ComputeError(adi.Solution, xDirect, n);
        }

        csv.AppendLine(string.Join(",",
            n,
            runs,
            Mean(adiIter).ToString("F2"),
            Std(adiIter).ToString("F2"),
            Mean(adiTime).ToString("F3"),
            Std(adiTime).ToString("F3"),
            Mean(directTime).ToString("F3"),
            Std(directTime).ToString("F3"),
            Mean(adiErr).ToString("E3"),
            Mean(directErr).ToString("E3"),
            Mean(crossErr).ToString("E3")));

        Console.WriteLine($"  n={n}: ADI={Mean(adiTime):F3} ms, direct={Mean(directTime):F3} ms");
    }

    File.WriteAllText(Path.Combine(outputDir, "adi_vs_direct.csv"), csv.ToString());
}

static (double[,], double[,], double[,], double[,]) GenerateTestProblem(int n, int seed)
{
    var rng = new Random(seed);
    var a = new double[n, n];
    var b = new double[n, n];
    var xExact = new double[n, n];

    for (var i = 0; i < n; i++)
    {
        var rowSumA = 0.0;
        var rowSumB = 0.0;

        for (var j = 0; j < n; j++)
        {
            a[i, j] = rng.NextDouble() * 2.0 - 1.0;
            b[i, j] = rng.NextDouble() * 2.0 - 1.0;

            if (i != j)
            {
                rowSumA += Math.Abs(a[i, j]);
                rowSumB += Math.Abs(b[i, j]);
            }

            xExact[i, j] = rng.NextDouble() * 2.0 - 1.0;
        }

        a[i, i] = rowSumA + 1.0 + rng.NextDouble();
        b[i, i] = rowSumB + 1.0 + rng.NextDouble();
    }

    var c = new double[n, n];
    for (var i = 0; i < n; i++)
    {
        for (var j = 0; j < n; j++)
        {
            var ax = 0.0;
            var xb = 0.0;
            for (var k = 0; k < n; k++)
            {
                ax += a[i, k] * xExact[k, j];
                xb += xExact[i, k] * b[k, j];
            }
            c[i, j] = ax + xb;
        }
    }

    return (a, b, c, xExact);
}

static (double minBound, double maxBound) GershgorinBounds(double[,] m)
{
    var n = m.GetLength(0);
    var minBound = double.PositiveInfinity;
    var maxBound = double.NegativeInfinity;

    for (var i = 0; i < n; i++)
    {
        var radius = 0.0;
        for (var j = 0; j < n; j++)
        {
            if (i != j)
            {
                radius += Math.Abs(m[i, j]);
            }
        }

        var left = m[i, i] - radius;
        var right = m[i, i] + radius;
        minBound = Math.Min(minBound, left);
        maxBound = Math.Max(maxBound, right);
    }

    minBound = Math.Max(1e-3, minBound);
    maxBound = Math.Max(minBound + 1e-3, maxBound);
    return (minBound, maxBound);
}

static double[] GenerateUniformShifts(int n, double scale)
{
    var count = Math.Max(8, (int)(Math.Log2(n + 1) * 6));
    var shifts = new double[count];

    for (var i = 0; i < count; i++)
    {
        shifts[i] = scale * (0.25 + 2.0 * i / Math.Max(1, count - 1));
    }

    return shifts;
}

static double[] GenerateLogShiftsFromBounds(double[,] matrix, int n)
{
    var (minB, maxB) = GershgorinBounds(matrix);
    var count = Math.Max(8, (int)(Math.Log2(n + 1) * 6));
    var shifts = new double[count];

    for (var i = 0; i < count; i++)
    {
        var t = (double)i / (count - 1);
        shifts[i] = minB * Math.Pow(maxB / minB, t);
    }

    return shifts;
}

static double[] GenerateSpectralAdaptiveShifts(double[,] matrix, int n)
{
    var (minB, maxB) = GershgorinBounds(matrix);
    var count = Math.Max(10, (int)(Math.Log2(n + 1) * 7));
    var shifts = new double[count];

    var spread = maxB / minB;
    for (var i = 0; i < count; i++)
    {
        var theta = (i + 0.5) * Math.PI / count;
        var mapped = 0.5 * (maxB + minB) + 0.5 * (maxB - minB) * Math.Cos(theta);
        shifts[i] = Math.Max(1e-3, mapped / Math.Sqrt(spread));
    }

    Array.Sort(shifts);
    return shifts;
}

static double[,] SolveSylvesterDirect(double[,] a, double[,] b, double[,] c)
{
    var n = a.GetLength(0);
    var nn = n * n;
    var k = new double[nn, nn];
    var rhs = new double[nn];

    int Idx(int i, int j) => i * n + j;

    for (var i = 0; i < n; i++)
    {
        for (var j = 0; j < n; j++)
        {
            var row = Idx(i, j);
            rhs[row] = c[i, j];

            for (var t = 0; t < n; t++)
            {
                k[row, Idx(t, j)] += a[i, t];
                k[row, Idx(i, t)] += b[t, j];
            }
        }
    }

    var sol = SolveLinearSystem(k, rhs);
    var x = new double[n, n];
    for (var i = 0; i < n; i++)
    {
        for (var j = 0; j < n; j++)
        {
            x[i, j] = sol[Idx(i, j)];
        }
    }

    return x;
}

static double[] SolveLinearSystem(double[,] a, double[] b)
{
    var n = b.Length;
    var m = (double[,])a.Clone();
    var x = (double[])b.Clone();

    for (var k = 0; k < n; k++)
    {
        var pivot = k;
        var pivotAbs = Math.Abs(m[k, k]);
        for (var i = k + 1; i < n; i++)
        {
            var cand = Math.Abs(m[i, k]);
            if (cand > pivotAbs)
            {
                pivotAbs = cand;
                pivot = i;
            }
        }

        if (pivotAbs < 1e-14)
        {
            throw new InvalidOperationException("Direct solver: singular system.");
        }

        if (pivot != k)
        {
            for (var j = k; j < n; j++)
            {
                var tmp = m[k, j];
                m[k, j] = m[pivot, j];
                m[pivot, j] = tmp;
            }
            var bx = x[k];
            x[k] = x[pivot];
            x[pivot] = bx;
        }

        for (var i = k + 1; i < n; i++)
        {
            var factor = m[i, k] / m[k, k];
            m[i, k] = 0.0;
            for (var j = k + 1; j < n; j++)
            {
                m[i, j] -= factor * m[k, j];
            }
            x[i] -= factor * x[k];
        }
    }

    for (var i = n - 1; i >= 0; i--)
    {
        var sum = x[i];
        for (var j = i + 1; j < n; j++)
        {
            sum -= m[i, j] * x[j];
        }
        x[i] = sum / m[i, i];
    }

    return x;
}

static double ComputeError(double[,] exact, double[,] approx, int n)
{
    var sum = 0.0;
    for (var i = 0; i < n; i++)
    {
        for (var j = 0; j < n; j++)
        {
            var d = exact[i, j] - approx[i, j];
            sum += d * d;
        }
    }
    return Math.Sqrt(sum);
}

static double Mean(double[] values) => values.Average();

static double Std(double[] values)
{
    if (values.Length <= 1)
    {
        return 0.0;
    }

    var mean = Mean(values);
    var sum = values.Sum(v => (v - mean) * (v - mean));
    return Math.Sqrt(sum / (values.Length - 1));
}
