using LNU.NMMPH.API.Models;

namespace LNU.NMMPH.API.Interface
{
    public interface IMethodsService
    {
        Task<object> Execute(Method method, IFormFile file);
    }
}
