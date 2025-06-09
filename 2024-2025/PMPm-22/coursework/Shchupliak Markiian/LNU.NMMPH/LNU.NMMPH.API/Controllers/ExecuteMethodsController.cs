using Microsoft.AspNetCore.Mvc;

using LNU.NMMPH.API.Controllers.Models;
using LNU.NMMPH.API.Interface;

namespace LNU.NMMPH.API.Controllers
{
    [Route("api/execute-methods")]
    [ApiController]
    public class ExecuteMethodsController : ControllerBase
    {
        private readonly IMethodsService _methodsService;

        public ExecuteMethodsController(IMethodsService methodsService)
            => _methodsService = methodsService;

        [HttpPost]
        public async Task<IActionResult> UploadFile([FromForm] PostExecuteMethodRequest request)
        {
            if (request.File == null || request.File.Length == 0)
                return BadRequest("No file uploaded.");

            if (!Path.GetExtension(request.File.FileName).Equals(".csx", StringComparison.CurrentCultureIgnoreCase))
                return BadRequest("Only .csx files are allowed.");

            if (!request.Method.HasValue)
                return BadRequest("No method specified.");

            double result = await _methodsService.Execute(request.Method.Value, request.File);

            return Ok(result);
        }
    }
}