using LNU.NMMPH.API.Models.Results;
using LNU.NMMPH.API.Models;

namespace LNU.NMMPH.API.Interface.Methods
{
    public interface IPoissonMethod
    {
        Task<Result<PoissonComparisonResult>> ExecuteStudent(string code);
    }
}
