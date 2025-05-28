using LNU.NMMPH.API.Models;

namespace LNU.NMMPH.API.Interface
{
    public interface IMethodsService
    {
        Task<double> Execute(Method method, IFormFile file);
    }
}
