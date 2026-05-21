using System;

namespace SparseCourseworkRunner;

public sealed class CsrMatrix
{
    public CsrMatrix(int size, int[] rowPointers, int[] columnIndices, double[] values)
    {
        Size = size;
        RowPointers = rowPointers;
        ColumnIndices = columnIndices;
        Values = values;
    }

    public int Size { get; }

    public int[] RowPointers { get; }

    public int[] ColumnIndices { get; }

    public double[] Values { get; }

    public int NonZeros => Values.Length;

    public void Multiply(ReadOnlySpan<double> x, Span<double> y)
    {
        if (x.Length != Size)
        {
            throw new ArgumentException("Input vector length mismatch.", nameof(x));
        }

        if (y.Length != Size)
        {
            throw new ArgumentException("Output vector length mismatch.", nameof(y));
        }

        for (var i = 0; i < Size; i++)
        {
            double sum = 0.0;
            for (var p = RowPointers[i]; p < RowPointers[i + 1]; p++)
            {
                sum += Values[p] * x[ColumnIndices[p]];
            }

            y[i] = sum;
        }
    }

    public double[] CloneValues() => (double[])Values.Clone();
}
