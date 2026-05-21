using SylvesterApi.Contracts;
using SylvesterSolver.FSharp;

namespace SylvesterApi.Services;

public sealed class SylvesterSolverService : ISylvesterSolverService
{
    public SolveSylvesterResponse Solve(SolveSylvesterRequest request)
    {
        var a = ToRectangular(request.A);
        var b = ToRectangular(request.B);
        var c = ToRectangular(request.C);

        var result = SylvesterAdiSolver.Solve(
            a,
            b,
            c,
            request.PShifts,
            request.QShifts,
            request.Tolerance,
            request.MaxIterations);

        return new SolveSylvesterResponse
        {
            Solution = ToJagged(result.Solution),
            ResidualNorm = result.ResidualNorm,
            Iterations = result.Iterations,
            Converged = result.Converged,
            ResidualHistory = result.ResidualHistory
        };
    }

    private static double[,] ToRectangular(double[][] matrix)
    {
        var rows = matrix.Length;
        var cols = matrix[0].Length;
        var result = new double[rows, cols];

        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                result[i, j] = matrix[i][j];
            }
        }

        return result;
    }

    private static double[][] ToJagged(double[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var result = new double[rows][];

        for (var i = 0; i < rows; i++)
        {
            result[i] = new double[cols];
            for (var j = 0; j < cols; j++)
            {
                result[i][j] = matrix[i, j];
            }
        }

        return result;
    }
}
