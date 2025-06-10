namespace MonteCarloWeb.Models;

public class IntegrationRequest
{
    public int[]? NArray { get; set; }
    public double LowerBound { get; set; }
    public double UpperBound { get; set; }
    public int Dimensions { get; set; }
    public string? Function { get; set; }
}
