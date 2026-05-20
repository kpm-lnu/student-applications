using System;

namespace SparseCourseworkRunner;

public sealed class PoissonProblem
{
    private PoissonProblem(int gridSize, CsrMatrix matrix, double[] rightHandSide, double[] exactSolution)
    {
        GridSize = gridSize;
        Matrix = matrix;
        RightHandSide = rightHandSide;
        ExactSolution = exactSolution;
    }

    public int GridSize { get; }

    public int Size => Matrix.Size;

    public CsrMatrix Matrix { get; }

    public double[] RightHandSide { get; }

    public double[] ExactSolution { get; }

    public static PoissonProblem Create(int n)
    {
        if (n <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(n), "Grid size n must be greater than 1.");
        }

        var size = checked(n * n);
        var nnz = checked(5 * size - 4 * n);

        var rowPointers = new int[size + 1];
        var columnIndices = new int[nnz];
        var values = new double[nnz];
        var rhs = new double[size];
        var exact = new double[size];

        var position = 0;
        var h = 1.0 / (n + 1.0);

        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var row = i * n + j;
                rowPointers[row] = position;

                if (i > 0)
                {
                    columnIndices[position] = row - n;
                    values[position] = -1.0;
                    position++;
                }

                if (j > 0)
                {
                    columnIndices[position] = row - 1;
                    values[position] = -1.0;
                    position++;
                }

                columnIndices[position] = row;
                values[position] = 4.0;
                position++;

                if (j < n - 1)
                {
                    columnIndices[position] = row + 1;
                    values[position] = -1.0;
                    position++;
                }

                if (i < n - 1)
                {
                    columnIndices[position] = row + n;
                    values[position] = -1.0;
                    position++;
                }

                var x = (i + 1) * h;
                var y = (j + 1) * h;
                var u = x * (1.0 - x) * y * (1.0 - y) * (1.0 + 0.5 * x + 0.25 * y);
                exact[row] = u;
            }
        }

        rowPointers[size] = position;

        if (position != nnz)
        {
            throw new InvalidOperationException($"CSR build mismatch. Expected {nnz}, got {position}.");
        }

        var matrix = new CsrMatrix(size, rowPointers, columnIndices, values);
        matrix.Multiply(exact, rhs);
        return new PoissonProblem(n, matrix, rhs, exact);
    }
}
