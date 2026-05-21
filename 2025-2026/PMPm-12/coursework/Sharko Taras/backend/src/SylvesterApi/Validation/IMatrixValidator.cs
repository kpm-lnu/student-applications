using SylvesterApi.Contracts;

namespace SylvesterApi.Validation;

public interface IMatrixValidator
{
    IReadOnlyCollection<string> Validate(SolveSylvesterRequest request);
}
