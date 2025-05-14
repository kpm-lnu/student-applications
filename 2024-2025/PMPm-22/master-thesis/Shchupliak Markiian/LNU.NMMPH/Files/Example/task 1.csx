using System;
using System.Collections.Generic;

// The input parameters from the main program: nx, ny, a, b, f
public static object Solve(int Nx, int Ny, double a, double b, double f)
    {
        int nNodes = Nx * Ny;
        double dx = a / (Nx - 1);
        double dy = b / (Ny - 1);

        var nodes = new List<(double x, double y)>();
        for (int j = 0; j < Ny; j++)
            for (int i = 0; i < Nx; i++)
                nodes.Add((i * dx, j * dy));

        var exact = GenerateExact(Nx, Ny, b, f);
        var u = new double[nNodes];
        var rhs = new double[nNodes];
        var A = new double[nNodes, nNodes];

        for (int j = 0; j < Ny; j++)
        {
            for (int i = 0; i < Nx; i++)
            {
                int idx = j * Nx + i;

                if (i == 0 || i == Nx - 1)
                {
                    A[idx, idx] = 1;
                    rhs[idx] = 0;
                    continue;
                }

                double coeff = 2 / (dx * dx) + 2 / (dy * dy);
                A[idx, idx] = coeff;
                A[idx, j * Nx + (i - 1)] = -1 / (dx * dx);
                A[idx, j * Nx + (i + 1)] = -1 / (dx * dx);

                if (j > 0)
                    A[idx, (j - 1) * Nx + i] = -1 / (dy * dy);
                else
                    rhs[idx] += f / dy;

                if (j < Ny - 1)
                    A[idx, (j + 1) * Nx + i] = -1 / (dy * dy);
                else
                    rhs[idx] -= f / dy;
            }
        }

        for (int it = 0; it < 5000; it++)
        {
            for (int i = 0; i < nNodes; i++)
            {
                double sum = 0;
                for (int j = 0; j < nNodes; j++)
                {
                    if (j != i) sum += A[i, j] * u[j];
                }
                u[i] = (rhs[i] - sum) / A[i, i];
            }
        }

        double errL2 = 0;

        for (int i = 0; i < nNodes; i++)
            errL2 += Math.Pow(u[i] - exact[i], 2);
        errL2 = Math.Sqrt(errL2 / nNodes);

        double errH1 = 0;

        for (int j = 1; j < Ny - 1; j++)
        {
            for (int i = 1; i < Nx - 1; i++)
            {
                int idx = j * Nx + i;
                int idxL = j * Nx + (i - 1);
                int idxR = j * Nx + (i + 1);
                int idxD = (j - 1) * Nx + i;
                int idxU = (j + 1) * Nx + i;

                double gradX_u = (u[idxR] - u[idxL]) / (2 * dx);
                double gradY_u = (u[idxU] - u[idxD]) / (2 * dy);
                double gradX_e = (exact[idxR] - exact[idxL]) / (2 * dx);
                double gradY_e = (exact[idxU] - exact[idxD]) / (2 * dy);

                errH1 += Math.Pow(gradX_u - gradX_e, 2) + Math.Pow(gradY_u - gradY_e, 2);
            }
        }
        errH1 = Math.Sqrt(errH1 * dx * dy / ((Nx - 2) * (Ny - 2)));

        return new
        {
            NumericResult = u,
            ExactResult = exact,
            L2Error = errL2,
            H1SemiError = errH1,
            Nodes = nodes
        };
	}

    private static double[] GenerateExact(int Nx, int Ny, double b, double f)
{
    int total = Nx * Ny;
    double dy = b / (Ny - 1);
    var result = new double[total];

    for (int i = 0; i < total; i++)
    {
        double y = (i / Nx) * dy;
        result[i] = 0.5 * f * y * (b - y);
    }

    return result;
}
	
object result = Solve(Nx, Ny, A, B, F);
return result;