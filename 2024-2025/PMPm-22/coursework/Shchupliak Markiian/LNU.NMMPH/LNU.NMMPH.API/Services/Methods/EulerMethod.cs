using Microsoft.CodeAnalysis.CSharp.Scripting;

using LNU.NMMPH.API.Interface.Methods;
using LNU.NMMPH.API.Models.Parameters;

namespace LNU.NMMPH.API.Services.Methods
{
    public class EulerMethod : IEulerMethod
    {
        private readonly EulerMethodParameters _parameters;

        public EulerMethod()
            => _parameters = new EulerMethodParameters();

        public async Task<double> ExecuteStudent(string code)
        {
            double[] eulerResult = await CSharpScript.EvaluateAsync<double[]>(code, globals: _parameters);

            double[] eulerExact = CorrectMethod();

            double eulerError = CalculateError(eulerResult, eulerExact);

            return eulerError;
        }

        private double[] CorrectMethod()
        {
            int steps = (int)((_parameters.TEnd - _parameters.T0) / _parameters.H);
            double[] results = new double[steps + 1];
            results[0] = _parameters.Y0;

            double t = _parameters.T0;
            double y = _parameters.Y0;

            for (int n = 0; n < steps; n++)
            {
                y = y + _parameters.H * _parameters.F(t, y);
                t = t + _parameters.H;

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