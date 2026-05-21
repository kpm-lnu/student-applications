namespace SylvesterApi.Contracts;

public sealed class SolveSylvesterRequest
{
    public required double[][] A { get; init; }
    public required double[][] B { get; init; }
    public required double[][] C { get; init; }
    public required double[] PShifts { get; init; }
    public required double[] QShifts { get; init; }
    public int MaxIterations { get; init; } = 500;
    public double Tolerance { get; init; } = 1e-8;
}
