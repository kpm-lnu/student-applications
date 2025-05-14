using LNU.NMMPH.API.Models;

namespace LNU.NMMPH.API.Interface.Methods
{
    public interface IRungeKuttaMethod
    {
        Task<Result<double>> ExecuteStudent(string code);
    }
}
