// integration_method.csx
using System;

// The input parameters from the main program: t0, y0, tEnd, h
public static double[] RungeKuttaMethod(double t0, double y0, double tEnd, double h, Func<double, double, double> f)
    {
        int steps = (int)((tEnd - t0) / h);
double[] results = new double[steps + 1];
results[0] = y0;

double t = t0;
double y = y0;

for (int n = 0; n < steps; n++)
{
	double k1 = h * f(t, y);
    double k2 = h * f(t + h / 2, y + k1 / 2);
    double k3 = h * f(t + h / 2, y + k2 / 2);
    double k4 = h * f(t + h, y + k3);

    y += (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    t += h;

    results[n + 1] = y;
}

return results;
    }
	
double[] result = RungeKuttaMethod(T0, Y0, TEnd, H, F);
return result;