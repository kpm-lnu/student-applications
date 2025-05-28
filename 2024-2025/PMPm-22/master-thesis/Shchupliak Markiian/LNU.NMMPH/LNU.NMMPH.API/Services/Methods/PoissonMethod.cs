using LNU.NMMPH.API.Interface;
using LNU.NMMPH.API.Interface.Methods;
using LNU.NMMPH.API.Models;
using LNU.NMMPH.API.Models.Parameters;
using LNU.NMMPH.API.Models.Results;

using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;

using Newtonsoft.Json;

using System;

namespace LNU.NMMPH.API.Services.Methods
{
    public class PoissonMethod : IPoissonMethod
    {
        private readonly PoissonMethodParameters _params;
        private readonly IGroqAiReviewService _aiReviewService;

        public PoissonMethod(IGroqAiReviewService aiReviewService)
        {
            _params = new PoissonMethodParameters();
            _aiReviewService = aiReviewService;
        }

        public async Task<Result<PoissonComparisonResult>> ExecuteStudent(string code)
        {
            object studentResult = await CSharpScript.EvaluateAsync<object>(
                code,
                ScriptOptions.Default,
                globals: _params
            );

            string json = JsonConvert.SerializeObject(studentResult);
            Poisson2DResult p = JsonConvert.DeserializeObject<Poisson2DResult>(json)!;

            double[] numerical = Solve().ExactResult;

            string reviewedCode = await _aiReviewService.ReviewCodeAsync(code, "Poisson 2D Method");

            return new Result<PoissonComparisonResult>
            {
                AiReview = reviewedCode,
                Value = new PoissonComparisonResult
                {
                    L2Error = CalculateL2Error(p.ExactResult, numerical),
                    MaxError = CalculateMaxError(p.ExactResult, numerical),
                    RelativeErrorPercent = CalculateRelativeErrorPercent(p.ExactResult, numerical)
                }
            };
        }

        private Poisson2DResult Solve()
        {
            int Nx = _params.Nx;
            int Ny = _params.Ny;
            int nNodes = Nx * Ny;
            double dx = _params.A / (Nx - 1);
            double dy = _params.B / (Ny - 1);
            double f = _params.F;

            var nodes = new List<(double x, double y)>();
            for (int j = 0; j < Ny; j++)
                for (int i = 0; i < Nx; i++)
                    nodes.Add((i * dx, j * dy));

            var exact = GenerateExact();
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

            return new Poisson2DResult
            {
                NumericResult = u,
                ExactResult = exact,
                L2Error = errL2,
                H1SemiError = errH1,
                Nodes = nodes
            };
        }

        private double[] GenerateExact()
        {
            int total = _params.Nx * _params.Ny;
            double dy = _params.B / (_params.Ny - 1);
            var result = new double[total];

            for (int i = 0; i < total; i++)
            {
                double y = (i / _params.Nx) * dy;
                result[i] = 0.5 * _params.F * y * (_params.B - y);
            }

            return result;
        }

        private double CalculateL2Error(double[] a, double[] b)
        {
            double sum = 0;

            for (int i = 0; i < a.Length; i++)
                sum += Math.Pow(a[i] - b[i], 2);

            return Math.Sqrt(sum / a.Length);
        }

        private double CalculateMaxError(double[] a, double[] b)
        {
            double max = 0;

            for (int i = 0; i < a.Length; i++)
                max = Math.Max(max, Math.Abs(a[i] - b[i]));

            return max;
        }

        private double CalculateRelativeErrorPercent(double[] approx, double[] reference)
        {
            double sumNum = 0, sumDen = 0;

            for (int i = 0; i < approx.Length; i++)
            {
                sumNum += Math.Abs(approx[i] - reference[i]);
                sumDen += Math.Abs(reference[i]);
            }

            return (sumDen == 0) ? 0 : ((sumNum / sumDen) + 1) * 100;
        }
    }
}
