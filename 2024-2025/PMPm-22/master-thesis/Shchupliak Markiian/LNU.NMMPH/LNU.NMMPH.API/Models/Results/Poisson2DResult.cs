namespace LNU.NMMPH.API.Models.Results
{
    public class Poisson2DResult
    {
        public double[] NumericResult { get; set; } = [];
        public double[] ExactResult { get; set; } = [];
        public double L2Error { get; set; }
        public double H1SemiError { get; set; }
        public List<(double x, double y)> Nodes { get; set; } = [];
    }
}
