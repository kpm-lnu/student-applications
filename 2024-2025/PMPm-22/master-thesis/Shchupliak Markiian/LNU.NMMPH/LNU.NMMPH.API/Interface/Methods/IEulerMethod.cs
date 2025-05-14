using LNU.NMMPH.API.Models;

namespace LNU.NMMPH.API.Interface.Methods
{
    public interface IEulerMethod
    {
        Task<Result<double>> ExecuteStudent(string code);
    }
}