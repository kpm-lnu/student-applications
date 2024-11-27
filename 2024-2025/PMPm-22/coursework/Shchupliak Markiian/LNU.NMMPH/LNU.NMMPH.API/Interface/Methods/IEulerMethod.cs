
namespace LNU.NMMPH.API.Interface.Methods
{
    public interface IEulerMethod
    {
        Task<double> ExecuteStudent(string eulerCode);
    }
}