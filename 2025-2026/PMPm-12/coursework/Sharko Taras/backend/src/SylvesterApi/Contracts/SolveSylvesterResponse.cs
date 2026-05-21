namespace SylvesterApi.Contracts;

public sealed class SolveSylvesterResponse
{
    public required double[][] Solution { get; init; }
    public required double ResidualNorm { get; init; }
    public required int Iterations { get; init; }
    public required bool Converged { get; init; }
    public required double[] ResidualHistory { get; init; }
}
