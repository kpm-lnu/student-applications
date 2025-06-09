using System;

// The input parameters from the main program: t0, y0, tEnd, h, f
public static double[] EulerMethod(double t0, double y0, double tEnd, double h, Func<double, double, double> f)
    {
        int steps = (int)((tEnd - t0) / h);
		double[] results = new double[steps + 1];
		results[0] = y0;
		
		double t = t0;
		double y = y0;
		
		for (int n = 0; n < steps; n++)
		{
			y = y + h * f(t, y);
			t = t + h;
		
			results[n + 1] = y;
		}

		return results;
    }
	
double[] result = EulerMethod(T0, Y0, TEnd, H, F);
return result;