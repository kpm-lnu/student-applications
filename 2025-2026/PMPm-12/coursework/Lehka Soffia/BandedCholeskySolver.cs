using System;

namespace SparseCourseworkRunner;

/// <summary>
/// Band Cholesky factorization A = L * L^T for symmetric positive definite banded matrices.
/// Storage: lband[i, k] = L[i, i-k] for k = 0..halfBw.
/// Fill-in stays within the original band — no extra memory needed.
/// Time:  O(N * halfBw^2),  Memory: O(N * halfBw).
/// </summary>
public sealed class BandedCholeskySolver
{
    private readonly int _n;
    private readonly int _halfBw;
    private readonly double[,] _lband;

    public int NonZerosL => CountNonZerosL();

    public static BandedCholeskySolver Factorize(CsrMatrix matrix, int halfBandwidth)
    {
        var n = matrix.Size;
        var bw = halfBandwidth;
        var lband = new double[n, bw + 1];

        // Copy lower triangular band of A into lband
        for (var i = 0; i < n; i++)
        {
            for (var p = matrix.RowPointers[i]; p < matrix.RowPointers[i + 1]; p++)
            {
                var j = matrix.ColumnIndices[p];
                if (j <= i && i - j <= bw)
                {
                    lband[i, i - j] = matrix.Values[p];
                }
            }
        }

        // Cholesky-Banachiewicz algorithm for banded SPD matrix
        for (var i = 0; i < n; i++)
        {
            // Diagonal: L[i,i] = sqrt(A[i,i] - sum_{j<i} L[i,j]^2)
            var diag = lband[i, 0];
            var jStart = Math.Max(0, i - bw);
            for (var j = jStart; j < i; j++)
            {
                var lij = lband[i, i - j];
                diag -= lij * lij;
            }

            if (diag <= 0.0)
                throw new InvalidOperationException($"Matrix not positive definite at row {i} (diag={diag}).");

            lband[i, 0] = Math.Sqrt(diag);
            var lii = lband[i, 0];

            // Sub-diagonal: L[k,i] = (A[k,i] - sum_j L[k,j]*L[i,j]) / L[i,i]
            var kEnd = Math.Min(i + bw, n - 1);
            for (var k = i + 1; k <= kEnd; k++)
            {
                var val = lband[k, k - i]; // A[k,i]
                var jj = Math.Max(Math.Max(0, i - bw), k - bw);
                for (var j = jj; j < i; j++)
                {
                    val -= lband[k, k - j] * lband[i, i - j];
                }

                lband[k, k - i] = val / lii;
            }
        }

        return new BandedCholeskySolver(n, bw, lband);
    }

    private BandedCholeskySolver(int n, int halfBw, double[,] lband)
    {
        _n = n;
        _halfBw = halfBw;
        _lband = lband;
    }

    /// <summary>Solve A*x = b in-place: overwrites rhs with the solution.</summary>
    public void Solve(double[] rhs, double[] solution)
    {
        if (rhs.Length != _n || solution.Length != _n)
            throw new ArgumentException("Vector length mismatch.");

        // Forward substitution: L * y = b
        var y = new double[_n];
        for (var i = 0; i < _n; i++)
        {
            var sum = rhs[i];
            var jStart = Math.Max(0, i - _halfBw);
            for (var j = jStart; j < i; j++)
            {
                sum -= _lband[i, i - j] * y[j];
            }

            y[i] = sum / _lband[i, 0];
        }

        // Back substitution: L^T * x = y
        for (var i = _n - 1; i >= 0; i--)
        {
            var sum = y[i];
            var jEnd = Math.Min(i + _halfBw, _n - 1);
            for (var j = i + 1; j <= jEnd; j++)
            {
                sum -= _lband[j, j - i] * solution[j];
            }

            solution[i] = sum / _lband[i, 0];
        }
    }

    private int CountNonZerosL()
    {
        var count = 0;
        for (var i = 0; i < _n; i++)
        {
            for (var k = 0; k <= Math.Min(_halfBw, i); k++)
            {
                count++;
            }
        }

        return count;
    }
}
