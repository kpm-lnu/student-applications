using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.Services;

namespace UniversityBookings.Controllers;

public record ChatRequest(ChatMessageDto[] History, string Message);
public record ChatResponse(string Text, bool AppointmentChanged);

[ApiController]
[Route("api/chat")]
[Authorize]
public class ChatController(AppDbContext db, GeminiService geminiService) : ControllerBase
{
    [HttpPost]
    public async Task<ActionResult<ChatResponse>> Chat([FromBody] ChatRequest req)
    {
        var oid = User.FindFirst("oid")?.Value
               ?? User.FindFirst("http://schemas.microsoft.com/identity/claims/objectidentifier")?.Value;
        var user = await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == oid);
        if (user is null) return Unauthorized();

        try
        {
            var (text, appointmentChanged) = await geminiService.ChatAsync(user, req.History, req.Message);
            return Ok(new ChatResponse(text, appointmentChanged));
        }
        catch (HttpRequestException ex)
        {
            return StatusCode(502, new { message = "AI service unavailable.", detail = ex.Message });
        }
        catch (InvalidOperationException ex)
        {
            return StatusCode(503, new { message = ex.Message });
        }
    }
}
