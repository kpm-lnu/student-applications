using SylvesterApi.Contracts;

namespace SylvesterApi.Validation;

public sealed class MatrixValidator : IMatrixValidator
{
    public IReadOnlyCollection<string> Validate(SolveSylvesterRequest request)
    {
        var errors = new List<string>();

        if (!IsSquare(request.A, out var m))
        {
            errors.Add("Matrix A must be a non-empty square matrix.");
        }

        if (!IsSquare(request.B, out var n))
        {
            errors.Add("Matrix B must be a non-empty square matrix.");
        }

        if (!IsRectangular(request.C, out var cRows, out var cCols))
        {
            errors.Add("Matrix C must be rectangular and non-empty.");
        }
        else if (errors.Count == 0 && (cRows != m || cCols != n))
        {
            errors.Add("Matrix C dimensions must be m x n where A is m x m and B is n x n.");
        }

        if (request.PShifts.Length == 0)
        {
            errors.Add("PShifts cannot be empty.");
        }

        if (request.QShifts.Length == 0)
        {
            errors.Add("QShifts cannot be empty.");
        }

        if (request.MaxIterations <= 0)
        {
            errors.Add("MaxIterations must be positive.");
        }

        if (request.Tolerance <= 0)
        {
            errors.Add("Tolerance must be positive.");
        }

        return errors;
    }

    private static bool IsSquare(double[][] matrix, out int n)
    {
        n = 0;
        if (!IsRectangular(matrix, out var rows, out var cols))
        {
            return false;
        }

        if (rows != cols)
        {
            return false;
        }

        n = rows;
        return true;
    }

    private static bool IsRectangular(double[][] matrix, out int rows, out int cols)
    {
        rows = matrix.Length;
        cols = rows == 0 ? 0 : matrix[0].Length;

        if (rows == 0 || cols == 0)
        {
            return false;
        }

        var expectedCols = cols;
        return matrix.All(r => r.Length == expectedCols);
    }
}
