
using Microsoft.AspNetCore.Mvc;
using MonteCarloWeb.Models;
using MonteCarloWeb.Services;

namespace MonteCarloWeb.Controllers;

[ApiController]
[Route("api/[controller]")]
public class IntegrationController : ControllerBase
{
    private readonly IntegrationService _service;

    public IntegrationController(IntegrationService service)
    {
        _service = service;
    }

    [HttpPost]
    public async Task<ActionResult<List<IntegrationResult>>> Post(
        [FromBody] IntegrationRequest request,
        CancellationToken cancellationToken)
    {
        try
        {
            var result = await Task.Run(() => _service.Integrate(request), cancellationToken);
            return Ok(result);
        }
        catch (OperationCanceledException)
        {
            return StatusCode(499, "Обчислення зупинено користувачем.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
            return StatusCode(500, "Серверна помилка: " + ex.Message);
        }
    }
}