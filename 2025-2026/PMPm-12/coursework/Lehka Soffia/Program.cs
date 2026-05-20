using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;

namespace SparseCourseworkRunner;

internal static class Program
{
	private static readonly int[] DefaultFillInSizes = [10, 20, 50, 100, 200, 500];
	private static readonly int[] DefaultBenchmarkSizes = [50, 100, 200, 500, 1000];

	private static int Main(string[] args)
	{
		var options = RunnerOptions.Parse(args);
		if (options.ShowHelp)
		{
			PrintHelp();
			return 0;
		}

		Console.WriteLine("Sparse SLAR coursework benchmark (.NET + C#)");
		Console.WriteLine($"Repeats per method: {options.Repeats}");
		Console.WriteLine($"Tolerance: {options.Tolerance.ToString("0.0e+0", CultureInfo.InvariantCulture)}");
		Console.WriteLine($"Max iterations: {options.MaxIterations}");
		Console.WriteLine($"LU max n: {options.LuMaxN}");
		Console.WriteLine($"Benchmark max n: {options.BenchMaxN}");
		Console.WriteLine();

		var fillInRows = new List<FillInRow>();
		foreach (var n in options.FillInSizes)
		{
			Console.WriteLine($"[Fill-in] n={n}");
			var problem = PoissonProblem.Create(n);
			fillInRows.Add(ComputeFillInRow(problem, options));
		}

		Console.WriteLine();

		var timingRows = new List<TimingRow>();
		foreach (var n in options.BenchmarkSizes)
		{
			if (n > options.BenchMaxN)
			{
				Console.WriteLine($"[Timing] n={n} (skipped: above bench-max-n={options.BenchMaxN})");
				timingRows.Add(CreateSkippedTimingRow(n));
				continue;
			}

			Console.WriteLine($"[Timing] n={n}");
			var problem = PoissonProblem.Create(n);
			timingRows.Add(ComputeTimingRow(problem, options));
		}

		PrintLatexFillInRows(fillInRows);
		PrintLatexTimingRows(timingRows);
		PrintLatexMemoryRows(timingRows);
		PrintLatexIterationRows(timingRows);
		PrintAccuracySummary(timingRows);

		return 0;
	}

	private static FillInRow ComputeFillInRow(PoissonProblem problem, RunnerOptions options)
	{
		var n = problem.GridSize;
		var nnzA = problem.Matrix.NonZeros;

		if (options.SkipLu)
		{
			return new FillInRow(n, problem.Size, nnzA, null, null, "skipped");
		}

		if (n > options.LuMaxN)
		{
			return new FillInRow(n, problem.Size, nnzA, null, null, "lu-limit");
		}

		try
		{
			var chol = BandedCholeskySolver.Factorize(problem.Matrix, n);
			var nnzL = chol.NonZerosL;
			var nnzLPlusU = 2 * nnzL - problem.Size;
			var fillFactor = (double)nnzLPlusU / nnzA;
			return new FillInRow(n, problem.Size, nnzA, nnzLPlusU, fillFactor, null);
		}
		catch (OutOfMemoryException)
		{
			return new FillInRow(n, problem.Size, nnzA, null, null, "out-of-memory");
		}
		catch (Exception ex)
		{
			return new FillInRow(n, problem.Size, nnzA, null, null, ex.Message);
		}
	}

	private static TimingRow ComputeTimingRow(PoissonProblem problem, RunnerOptions options)
	{
		var luBenchmark = options.SkipLu || problem.GridSize > options.LuMaxN
			? TimedMethodResult.Failure("skipped")
			: BenchmarkMethod(options.Repeats, () => RunBandCholesky(problem));

		var bicgBenchmark = BenchmarkMethod(options.Repeats, () => RunBiCg(problem, options, useIlu: false));
		var bicgIluBenchmark = BenchmarkMethod(options.Repeats, () => RunBiCg(problem, options, useIlu: true));

		double? luMemoryMb = luBenchmark.Success && luBenchmark.ReferenceResult?.NnzLuFactor is int nnzL
			? EstimateBandCholeskyMemoryMb(problem.Size, problem.GridSize, nnzL)
			: null;

		var bicgIluMemoryMb = EstimateBiCgIluMemoryMb(problem.Size, problem.Matrix.NonZeros);

		return new TimingRow(
			problem.GridSize,
			problem.Size,
			luBenchmark,
			bicgBenchmark,
			bicgIluBenchmark,
			luMemoryMb,
			bicgIluMemoryMb);
	}

	private static TimingRow CreateSkippedTimingRow(int n)
	{
		var size = checked(n * n);
		var nnzA = checked(5 * size - 4 * n);
		var lu = TimedMethodResult.Failure("skipped");
		var bicg = TimedMethodResult.Failure("size-limit");

		return new TimingRow(
			n,
			size,
			lu,
			bicg,
			bicg,
			null,
			EstimateBiCgIluMemoryMb(size, nnzA));
	}

	private static MethodResult RunBandCholesky(PoissonProblem problem)
	{
		try
		{
			var chol = BandedCholeskySolver.Factorize(problem.Matrix, problem.GridSize);
			var x = new double[problem.Size];
			chol.Solve(problem.RightHandSide, x);

			var relResidual = ComputeRelativeResidual(problem.Matrix, x, problem.RightHandSide);
			var infError = ComputeInfinityError(x, problem.ExactSolution);
			var nnzL = chol.NonZerosL;

			return MethodResult.SuccessResult(relResidual, infError, 0, nnzL);
		}
		catch (OutOfMemoryException)
		{
			return MethodResult.FailureResult("out-of-memory");
		}
		catch (Exception ex)
		{
			return MethodResult.FailureResult(ex.Message);
		}
	}

	private static MethodResult RunBiCg(PoissonProblem problem, RunnerOptions options, bool useIlu)
	{
		try
		{
			IPreconditioner? preconditioner = useIlu ? new Ilu0Preconditioner(problem.Matrix) : null;

			var result = BiCgStabSolver.Solve(
				problem.Matrix,
				problem.RightHandSide,
				options.Tolerance,
				options.MaxIterations,
				preconditioner);

			if (!result.Converged)
			{
				return MethodResult.FailureResult("not-converged");
			}

			var infError = ComputeInfinityError(result.Solution, problem.ExactSolution);
			return MethodResult.SuccessResult(result.RelativeResidual, infError, result.Iterations, null);
		}
		catch (OutOfMemoryException)
		{
			return MethodResult.FailureResult("out-of-memory");
		}
		catch (Exception ex)
		{
			return MethodResult.FailureResult(ex.Message);
		}
	}

	private static TimedMethodResult BenchmarkMethod(int repeats, Func<MethodResult> run)
	{
		var samples = new List<double>(repeats);
		MethodResult? firstSuccess = null;

		for (var i = 0; i < repeats; i++)
		{
			GC.Collect();
			GC.WaitForPendingFinalizers();

			var sw = Stopwatch.StartNew();
			var result = run();
			sw.Stop();

			if (!result.Success)
			{
				return TimedMethodResult.Failure(result.Error ?? "method failed");
			}

			samples.Add(sw.Elapsed.TotalSeconds);
			firstSuccess ??= result;
		}

		return TimedMethodResult.SuccessResult(Median(samples), firstSuccess!);
	}

	private static double ComputeRelativeResidual(CsrMatrix matrix, double[] x, double[] b)
	{
		var ax = new double[matrix.Size];
		matrix.Multiply(x, ax);

		double residualSquared = 0.0;
		double bSquared = 0.0;

		for (var i = 0; i < matrix.Size; i++)
		{
			var ri = b[i] - ax[i];
			residualSquared += ri * ri;
			bSquared += b[i] * b[i];
		}

		var denom = bSquared > 0.0 ? Math.Sqrt(bSquared) : 1.0;
		return Math.Sqrt(residualSquared) / denom;
	}

	private static double ComputeInfinityError(double[] x, double[] exact)
	{
		var max = 0.0;
		for (var i = 0; i < x.Length; i++)
		{
			var diff = Math.Abs(x[i] - exact[i]);
			if (diff > max)
			{
				max = diff;
			}
		}

		return max;
	}

	// Band Cholesky memory: lband array (N × (halfBw+1)) × double + diagonal indices
	private static double EstimateBandCholeskyMemoryMb(int size, int halfBw, int nnzL)
	{
		var lbandBytes = (long)size * (halfBw + 1) * sizeof(double);
		return lbandBytes / (1024.0 * 1024.0);
	}

	private static double EstimateBiCgIluMemoryMb(int size, int nnzA)
	{
		var csrBytes = nnzA * (sizeof(double) + sizeof(int)) + (size + 1L) * sizeof(int);
		var vectorsBytes = 10L * size * sizeof(double);
		var totalBytes = 2L * csrBytes + vectorsBytes;
		return totalBytes / (1024.0 * 1024.0);
	}

	private static double Median(List<double> values)
	{
		values.Sort();
		var mid = values.Count / 2;
		return values.Count % 2 == 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
	}

	private static void PrintLatexFillInRows(IEnumerable<FillInRow> rows)
	{
		const string rowEnd = " \\\\";

		Console.WriteLine();
		Console.WriteLine("===== LaTeX rows: fill-in table =====");

		foreach (var row in rows)
		{
			if (row.NnzLuFactor is int nnzLu && row.FillFactor is double factor)
			{
				Console.WriteLine(
					$"{row.GridSize} & {FormatLatexInt(row.Size)} & {FormatLatexInt(row.NnzA)} & {FormatLatexInt(nnzLu)} & {FormatDecimal(factor, "0.0")}{rowEnd}");
			}
			else
			{
				Console.WriteLine(
					$"{row.GridSize} & {FormatLatexInt(row.Size)} & {FormatLatexInt(row.NnzA)} & -- & {FormatLuFailureForTable(row.Note)}{rowEnd}");
			}
		}
	}

	private static void PrintLatexTimingRows(IEnumerable<TimingRow> rows)
	{
		const string rowEnd = " \\\\";

		Console.WriteLine();
		Console.WriteLine("===== LaTeX rows: timing table =====");

		foreach (var row in rows)
		{
			Console.WriteLine(
				$"{row.GridSize} & {FormatLatexInt(row.Size)} & {FormatTimeOrFailure(row.Lu)} & {FormatTimeOrFailure(row.BiCg)} & {FormatTimeOrFailure(row.BiCgIlu)}{rowEnd}");
		}
	}

	private static void PrintLatexMemoryRows(IEnumerable<TimingRow> rows)
	{
		const string rowEnd = " \\\\";

		Console.WriteLine();
		Console.WriteLine("===== LaTeX rows: memory table =====");

		foreach (var row in rows)
		{
			var luMemText = row.LuMemoryMb.HasValue
				? $"\\approx {FormatDecimal(row.LuMemoryMb.Value, "0.#")}"
				: FormatLuFailureForTable(row.Lu.Error);

			Console.WriteLine(
				$"{row.GridSize} & {FormatLatexInt(row.Size)} & {luMemText} & \\approx {FormatDecimal(row.BiCgIluMemoryMb, "0.#")}{rowEnd}");
		}
	}

	private static void PrintLatexIterationRows(IEnumerable<TimingRow> rows)
	{
		const string rowEnd = " \\\\";

		Console.WriteLine();
		Console.WriteLine("===== LaTeX rows: iteration table =====");

		foreach (var row in rows)
		{
			var noPrec = row.BiCg.ReferenceResult?.Iterations.ToString(CultureInfo.InvariantCulture) ?? "--";
			var ilu = row.BiCgIlu.ReferenceResult?.Iterations.ToString(CultureInfo.InvariantCulture) ?? "--";

			Console.WriteLine($"{row.GridSize} & {FormatLatexInt(row.Size)} & {noPrec} & {ilu}{rowEnd}");
		}
	}

	private static void PrintAccuracySummary(IEnumerable<TimingRow> rows)
	{
		Console.WriteLine();
		Console.WriteLine("===== Accuracy summary =====");

		foreach (var row in rows)
		{
			var luError = row.Lu.ReferenceResult?.InfinityError;
			var bicgError = row.BiCg.ReferenceResult?.InfinityError;
			var bicgIluError = row.BiCgIlu.ReferenceResult?.InfinityError;

			Console.WriteLine(
				$"n={row.GridSize}: err_inf(Chol)={FormatMaybeDouble(luError)}, err_inf(BiCGSTAB)={FormatMaybeDouble(bicgError)}, err_inf(BiCGSTAB+ILU0)={FormatMaybeDouble(bicgIluError)}");
		}
	}

	private static string FormatTimeOrFailure(TimedMethodResult result)
	{
		if (result.Success && result.MedianSeconds is double time)
		{
			return FormatDecimal(time, "0.###");
		}

		return FormatLuFailureForTable(result.Error);
	}

	private static string FormatLuFailureForTable(string? error)
	{
		if (string.Equals(error, "out-of-memory", StringComparison.OrdinalIgnoreCase))
		{
			return "брак пам.";
		}

		if (string.Equals(error, "lu-limit", StringComparison.OrdinalIgnoreCase))
		{
			return "пропущено";
		}

		if (string.Equals(error, "skipped", StringComparison.OrdinalIgnoreCase))
		{
			return "пропущено";
		}

		if (string.Equals(error, "size-limit", StringComparison.OrdinalIgnoreCase))
		{
			return "--";
		}

		return "--";
	}

	private static string FormatLatexInt(int value)
	{
		var text = Math.Abs(value).ToString(CultureInfo.InvariantCulture);
		var parts = new List<string>();

		for (var i = text.Length; i > 0; i -= 3)
		{
			var start = Math.Max(0, i - 3);
			parts.Insert(0, text[start..i]);
		}

		var joined = string.Join("\\,", parts);
		return value < 0 ? $"-{joined}" : joined;
	}

	private static string FormatDecimal(double value, string format)
	{
		return value.ToString(format, CultureInfo.InvariantCulture).Replace('.', ',');
	}

	private static string FormatMaybeDouble(double? value)
	{
		return value.HasValue ? value.Value.ToString("0.00e+0", CultureInfo.InvariantCulture) : "--";
	}

	private static void PrintHelp()
	{
		Console.WriteLine("Usage:");
		Console.WriteLine("  dotnet run --project SparseCourseworkRunner -- [options]");
		Console.WriteLine();
		Console.WriteLine("Options:");
		Console.WriteLine("  --repeats=<int>       Number of repetitions per timing (default: 5)");
		Console.WriteLine("  --tol=<double>        Relative residual tolerance (default: 1e-6)");
		Console.WriteLine("  --max-iter=<int>      Max iterations for BiCGSTAB (default: 20000)");
		Console.WriteLine("  --lu-max-n=<int>      Skip direct solver for n above this limit (default: 200)");
		Console.WriteLine("  --bench-max-n=<int>   Skip timing runs for n above this limit (default: 500)");
		Console.WriteLine("  --fillin=a,b,c        Grid sizes for fill-in table");
		Console.WriteLine("  --bench=a,b,c         Grid sizes for timing/memory/iteration tables");
		Console.WriteLine("  --skip-lu             Skip direct solver runs");
		Console.WriteLine("  --help                Show this help");
	}

	private sealed record FillInRow(
		int GridSize,
		int Size,
		int NnzA,
		int? NnzLuFactor,
		double? FillFactor,
		string? Note);

	private sealed record TimingRow(
		int GridSize,
		int Size,
		TimedMethodResult Lu,
		TimedMethodResult BiCg,
		TimedMethodResult BiCgIlu,
		double? LuMemoryMb,
		double BiCgIluMemoryMb);

	private sealed record MethodResult(
		bool Success,
		string? Error,
		double RelativeResidual,
		double InfinityError,
		int Iterations,
		int? NnzLuFactor)
	{
		public static MethodResult SuccessResult(double relativeResidual, double infinityError, int iterations, int? nnzLuFactor)
		{
			return new MethodResult(true, null, relativeResidual, infinityError, iterations, nnzLuFactor);
		}

		public static MethodResult FailureResult(string error)
		{
			return new MethodResult(false, error, double.NaN, double.NaN, 0, null);
		}
	}

	private sealed record TimedMethodResult(
		bool Success,
		string? Error,
		double? MedianSeconds,
		MethodResult? ReferenceResult)
	{
		public static TimedMethodResult SuccessResult(double medianSeconds, MethodResult referenceResult)
		{
			return new TimedMethodResult(true, null, medianSeconds, referenceResult);
		}

		public static TimedMethodResult Failure(string error)
		{
			return new TimedMethodResult(false, error, null, null);
		}
	}

	private sealed record RunnerOptions(
		int Repeats,
		double Tolerance,
		int MaxIterations,
		int LuMaxN,
		int BenchMaxN,
		int[] FillInSizes,
		int[] BenchmarkSizes,
		bool SkipLu,
		bool ShowHelp)
	{
		public static RunnerOptions Parse(string[] args)
		{
			var repeats = 5;
			var tolerance = 1e-6;
			var maxIterations = 20_000;
			var luMaxN = 200;
			var benchMaxN = 500;
			var fillInSizes = DefaultFillInSizes;
			var benchmarkSizes = DefaultBenchmarkSizes;
			var skipLu = false;
			var showHelp = false;

			foreach (var arg in args)
			{
				if (arg.Equals("--help", StringComparison.OrdinalIgnoreCase))
				{
					showHelp = true;
				}
				else if (arg.Equals("--skip-lu", StringComparison.OrdinalIgnoreCase))
				{
					skipLu = true;
				}
				else if (arg.StartsWith("--repeats=", StringComparison.OrdinalIgnoreCase))
				{
					repeats = ParsePositiveInt(arg, "--repeats=");
				}
				else if (arg.StartsWith("--tol=", StringComparison.OrdinalIgnoreCase))
				{
					tolerance = ParsePositiveDouble(arg, "--tol=");
				}
				else if (arg.StartsWith("--max-iter=", StringComparison.OrdinalIgnoreCase))
				{
					maxIterations = ParsePositiveInt(arg, "--max-iter=");
				}
				else if (arg.StartsWith("--lu-max-n=", StringComparison.OrdinalIgnoreCase))
				{
					luMaxN = ParsePositiveInt(arg, "--lu-max-n=");
				}
				else if (arg.StartsWith("--bench-max-n=", StringComparison.OrdinalIgnoreCase))
				{
					benchMaxN = ParsePositiveInt(arg, "--bench-max-n=");
				}
				else if (arg.StartsWith("--fillin=", StringComparison.OrdinalIgnoreCase))
				{
					fillInSizes = ParseIntList(arg, "--fillin=");
				}
				else if (arg.StartsWith("--bench=", StringComparison.OrdinalIgnoreCase))
				{
					benchmarkSizes = ParseIntList(arg, "--bench=");
				}
				else
				{
					throw new ArgumentException($"Unknown argument: {arg}");
				}
			}

			return new RunnerOptions(repeats, tolerance, maxIterations, luMaxN, benchMaxN, fillInSizes, benchmarkSizes, skipLu, showHelp);
		}

		private static int ParsePositiveInt(string argument, string prefix)
		{
			var text = argument[prefix.Length..];
			if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) || value <= 0)
			{
				throw new ArgumentException($"Invalid integer value in {argument}");
			}

			return value;
		}

		private static double ParsePositiveDouble(string argument, string prefix)
		{
			var text = argument[prefix.Length..];
			if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) || value <= 0.0)
			{
				throw new ArgumentException($"Invalid double value in {argument}");
			}

			return value;
		}

		private static int[] ParseIntList(string argument, string prefix)
		{
			var text = argument[prefix.Length..];
			var values = text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
				.Select(item => int.Parse(item, CultureInfo.InvariantCulture))
				.ToArray();

			if (values.Length == 0 || values.Any(v => v <= 1))
			{
				throw new ArgumentException($"Invalid integer list in {argument}");
			}

			return values;
		}
	}
}
