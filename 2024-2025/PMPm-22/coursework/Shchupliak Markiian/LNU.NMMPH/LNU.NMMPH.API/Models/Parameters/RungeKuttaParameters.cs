namespace LNU.NMMPH.API.Models.Parameters
{
    public class RungeKuttaParameters
    {
        public double T0 { get; }
        public double Y0 { get; }
        public double TEnd { get; }
        public double H { get; }
        public Func<double, double, double> F => (double t, double y) => -2 * y + 1;

        public RungeKuttaParameters()
        {
            T0 = 0;
            Y0 = 1;
            TEnd = 2;
            H = 0.1;
        }
    }
}
