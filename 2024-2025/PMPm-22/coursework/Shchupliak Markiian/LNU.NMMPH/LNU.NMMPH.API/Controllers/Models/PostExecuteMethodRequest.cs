using LNU.NMMPH.API.Models;

namespace LNU.NMMPH.API.Controllers.Models
{
    public class PostExecuteMethodRequest
    {
        public IFormFile? File { get; set; }
        public Method? Method { get; set; }
    }
}
