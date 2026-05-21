using System;

namespace SparseCourseworkRunner;

public sealed class Ilu0Preconditioner : IPreconditioner
{
    private const double PivotEpsilon = 1e-14;

    private readonly int _size;
    private readonly int[] _rowPointers;
    private readonly int[] _columnIndices;
    private readonly double[] _luValues;
    private readonly int[] _diagIndices;
    private readonly double[] _forwardBuffer;

    public Ilu0Preconditioner(CsrMatrix matrix)
    {
        _size = matrix.Size;
        _rowPointers = matrix.RowPointers;
        _columnIndices = matrix.ColumnIndices;
        _luValues = matrix.CloneValues();
        _diagIndices = LocateDiagonalIndices(_size, _rowPointers, _columnIndices);
        _forwardBuffer = new double[_size];

        FactorizeInPlace();
    }

    public void Apply(ReadOnlySpan<double> rhs, Span<double> destination)
    {
        if (rhs.Length != _size)
        {
            throw new ArgumentException("Right-hand side length mismatch.", nameof(rhs));
        }

        if (destination.Length != _size)
        {
            throw new ArgumentException("Destination length mismatch.", nameof(destination));
        }

        for (var i = 0; i < _size; i++)
        {
            var sum = rhs[i];
            var rowStart = _rowPointers[i];
            var rowEnd = _rowPointers[i + 1];

            for (var p = rowStart; p < rowEnd; p++)
            {
                var col = _columnIndices[p];
                if (col >= i)
                {
                    break;
                }

                sum -= _luValues[p] * _forwardBuffer[col];
            }

            _forwardBuffer[i] = sum;
        }

        for (var i = _size - 1; i >= 0; i--)
        {
            var sum = _forwardBuffer[i];
            var diagIndex = _diagIndices[i];
            var diag = _luValues[diagIndex];

            if (Math.Abs(diag) <= PivotEpsilon)
            {
                throw new InvalidOperationException($"Zero pivot in ILU(0) at row {i}.");
            }

            var rowEnd = _rowPointers[i + 1];
            for (var p = diagIndex + 1; p < rowEnd; p++)
            {
                sum -= _luValues[p] * destination[_columnIndices[p]];
            }

            destination[i] = sum / diag;
        }
    }

    private static int[] LocateDiagonalIndices(int size, int[] rowPointers, int[] columnIndices)
    {
        var diagIndices = new int[size];

        for (var i = 0; i < size; i++)
        {
            var found = false;
            for (var p = rowPointers[i]; p < rowPointers[i + 1]; p++)
            {
                if (columnIndices[p] == i)
                {
                    diagIndices[i] = p;
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                throw new InvalidOperationException($"Diagonal element missing at row {i}.");
            }
        }

        return diagIndices;
    }

    private void FactorizeInPlace()
    {
        for (var i = 0; i < _size; i++)
        {
            var rowStart = _rowPointers[i];
            var rowEnd = _rowPointers[i + 1];

            for (var p = rowStart; p < rowEnd; p++)
            {
                var j = _columnIndices[p];
                if (j >= i)
                {
                    break;
                }

                var diag = _luValues[_diagIndices[j]];
                if (Math.Abs(diag) <= PivotEpsilon)
                {
                    throw new InvalidOperationException($"Zero pivot in ILU(0) at row {j}.");
                }

                _luValues[p] /= diag;
                var lij = _luValues[p];

                var q = _diagIndices[j] + 1;
                var qEnd = _rowPointers[j + 1];
                var r = p + 1;

                while (r < rowEnd && q < qEnd)
                {
                    var colR = _columnIndices[r];
                    var colQ = _columnIndices[q];

                    if (colR == colQ)
                    {
                        _luValues[r] -= lij * _luValues[q];
                        r++;
                        q++;
                    }
                    else if (colR < colQ)
                    {
                        r++;
                    }
                    else
                    {
                        q++;
                    }
                }
            }
        }
    }
}
