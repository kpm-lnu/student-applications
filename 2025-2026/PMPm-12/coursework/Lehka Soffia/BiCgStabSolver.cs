using System;

namespace SparseCourseworkRunner;

public sealed record BiCgStabResult(bool Converged, int Iterations, double RelativeResidual, double[] Solution);

public static class BiCgStabSolver
{
    private const double BreakdownEpsilon = 1e-30;

    public static BiCgStabResult Solve(
        CsrMatrix matrix,
        double[] rightHandSide,
        double tolerance,
        int maxIterations,
        IPreconditioner? preconditioner = null)
    {
        var n = matrix.Size;
        if (rightHandSide.Length != n)
        {
            throw new ArgumentException("Right-hand side length mismatch.", nameof(rightHandSide));
        }

        if (maxIterations <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations), "maxIterations must be positive.");
        }

        if (tolerance <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(tolerance), "tolerance must be positive.");
        }

        var x = new double[n];
        var r = new double[n];
        var rHat = new double[n];
        var p = new double[n];
        var v = new double[n];
        var s = new double[n];
        var t = new double[n];
        var pHat = new double[n];
        var sHat = new double[n];

        Array.Copy(rightHandSide, r, n);
        Array.Copy(r, rHat, n);

        var bNorm = Norm2(rightHandSide);
        if (bNorm == 0.0)
        {
            bNorm = 1.0;
        }

        var initialResidual = Norm2(r) / bNorm;
        if (initialResidual <= tolerance)
        {
            return new BiCgStabResult(true, 0, initialResidual, x);
        }

        var rhoPrev = 1.0;
        var alpha = 1.0;
        var omega = 1.0;
        var relativeResidual = initialResidual;

        for (var iter = 1; iter <= maxIterations; iter++)
        {
            var rho = Dot(rHat, r);
            if (Math.Abs(rho) <= BreakdownEpsilon)
            {
                return new BiCgStabResult(false, iter - 1, relativeResidual, x);
            }

            if (iter == 1)
            {
                Array.Copy(r, p, n);
            }
            else
            {
                var beta = (rho / rhoPrev) * (alpha / omega);
                for (var i = 0; i < n; i++)
                {
                    p[i] = r[i] + beta * (p[i] - omega * v[i]);
                }
            }

            ApplyPreconditioner(preconditioner, p, pHat);

            matrix.Multiply(pHat, v);
            var denominator = Dot(rHat, v);
            if (Math.Abs(denominator) <= BreakdownEpsilon)
            {
                return new BiCgStabResult(false, iter - 1, relativeResidual, x);
            }

            alpha = rho / denominator;

            for (var i = 0; i < n; i++)
            {
                s[i] = r[i] - alpha * v[i];
            }

            var sNorm = Norm2(s);
            if (sNorm / bNorm <= tolerance)
            {
                for (var i = 0; i < n; i++)
                {
                    x[i] += alpha * pHat[i];
                }

                return new BiCgStabResult(true, iter, sNorm / bNorm, x);
            }

            ApplyPreconditioner(preconditioner, s, sHat);
            matrix.Multiply(sHat, t);

            var tt = Dot(t, t);
            if (Math.Abs(tt) <= BreakdownEpsilon)
            {
                return new BiCgStabResult(false, iter - 1, relativeResidual, x);
            }

            omega = Dot(t, s) / tt;
            if (Math.Abs(omega) <= BreakdownEpsilon)
            {
                return new BiCgStabResult(false, iter - 1, relativeResidual, x);
            }

            for (var i = 0; i < n; i++)
            {
                x[i] += alpha * pHat[i] + omega * sHat[i];
                r[i] = s[i] - omega * t[i];
            }

            relativeResidual = Norm2(r) / bNorm;
            if (relativeResidual <= tolerance)
            {
                return new BiCgStabResult(true, iter, relativeResidual, x);
            }

            rhoPrev = rho;
        }

        return new BiCgStabResult(false, maxIterations, relativeResidual, x);
    }

    private static void ApplyPreconditioner(IPreconditioner? preconditioner, ReadOnlySpan<double> rhs, Span<double> destination)
    {
        if (preconditioner is null)
        {
            rhs.CopyTo(destination);
            return;
        }

        preconditioner.Apply(rhs, destination);
    }

    private static double Dot(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        double sum = 0.0;
        for (var i = 0; i < x.Length; i++)
        {
            sum += x[i] * y[i];
        }

        return sum;
    }

    private static double Norm2(ReadOnlySpan<double> x)
    {
        return Math.Sqrt(Dot(x, x));
    }
}
