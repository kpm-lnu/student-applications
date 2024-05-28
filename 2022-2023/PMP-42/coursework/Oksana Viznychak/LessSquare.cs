using System;
using System.Collections.Generic;


namespace BoxCountingFractal.Logic
{
    public class LessSquare
    {
        public static void GetCoefficient(IList<Tuple<double, double>> points, out double k, out double b)
        {
            double n = points.Count;
            double Sx = 0; //сума всіх x
            double Sy = 0; //сума всіх y
            double Sxx = 0; //сума квадратів всіх x
            double Sxy = 0; //сума твору всіх x та y

            foreach (Tuple<double, double> point in points)
            {
                Sx += point.Item1;
                Sy += point.Item2;
                Sxx += point.Item1 * point.Item1;
                Sxy += point.Item1 * point.Item2;
            }

            k = (n * Sxy - Sy * Sx) / (n * Sxx - Sx * Sx);
            b = (Sxy * Sx - Sy * Sxx) / (Sx * Sx - n * Sxx);
        }
    }
}
