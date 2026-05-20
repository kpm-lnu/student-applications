using SylvesterApi.Contracts;

namespace SylvesterApi.Services;

public interface ISylvesterSolverService
{
    SolveSylvesterResponse Solve(SolveSylvesterRequest request);
}
