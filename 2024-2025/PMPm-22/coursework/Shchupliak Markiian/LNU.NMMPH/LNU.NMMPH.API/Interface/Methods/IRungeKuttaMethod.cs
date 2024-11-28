
namespace LNU.NMMPH.API.Interface.Methods
{
    public interface IRungeKuttaMethod
    {
        Task<double> ExecuteStudent(string code);
    }
}
