using Microsoft.CodeAnalysis.CSharp.Scripting;

using LNU.NMMPH.API.Interface.Methods;
using LNU.NMMPH.API.Models.Parameters;

namespace LNU.NMMPH.API.Services.Methods
{
    public class RungeKuttaMethod : IRungeKuttaMethod
    {
        private readonly RungeKuttaParameters _parameters;

        public RungeKuttaMethod()
            => _parameters = new RungeKuttaParameters();

        public async Task<double> ExecuteStudent(string code)
        {
            double[] rungeKuttaResult = await CSharpScript.EvaluateAsync<double[]>(code, globals: _parameters);

            double[] rungeKuttaExact = CurrectMethod();

            double rungeKuttaError = CalculateError(rungeKuttaResult, rungeKuttaExact);

            return rungeKuttaError;
        }

        private double[] CurrectMethod()
        {
            int steps = (int)((_parameters.TEnd - _parameters.T0) / _parameters.H);
            double[] results = new double[steps + 1];
            results[0] = _parameters.Y0;

            double t = _parameters.T0;
            double y = _parameters.Y0;

            for (int n = 0; n < steps; n++)
            {
                double k1 = _parameters.H * _parameters.F(t, y);
                double k2 = _parameters.H * _parameters.F(t + _parameters.H / 2, y + k1 / 2);
                double k3 = _parameters.H * _parameters.F(t + _parameters.H / 2, y + k2 / 2);
                double k4 = _parameters.H * _parameters.F(t + _parameters.H, y + k3);

                y += (k1 + 2 * k2 + 2 * k3 + k4) / 6;
                t += _parameters.H;

                results[n + 1] = y;
            }

            return results;
        }

        private static double CalculateError(double[] methodResults, double[] exactResults)
        {
            double totalError = 0;
            int n = methodResults.Length;

            for (int i = 0; i < n; i++)
                totalError += Math.Abs(methodResults[i] - exactResults[i]) / Math.Abs(exactResults[i]) * 100;

            return 100 - totalError / n;
        }
    }
}