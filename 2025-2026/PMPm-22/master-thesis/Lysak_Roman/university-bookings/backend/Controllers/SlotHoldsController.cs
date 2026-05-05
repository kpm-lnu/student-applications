using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using UniversityBookings.Data;
using UniversityBookings.Services;

namespace UniversityBookings.Controllers;

[ApiController]
[Route("api/slot-holds")]
[Authorize]
public class SlotHoldsController(AppDbContext db, SlotHoldService holdService) : ControllerBase
{
    private Guid CurrentUserId()
    {
        var oid = User.FindFirst("oid")?.Value
               ?? User.FindFirst("http://schemas.microsoft.com/identity/claims/objectidentifier")?.Value;
        var user = db.Users.FirstOrDefault(u => u.AzureObjectId == oid);
        return user?.Id ?? Guid.Empty;
    }

    [HttpPost]
    public async Task<IActionResult> Create([FromBody] CreateSlotHoldRequest req)
    {
        var userId = CurrentUserId();
        if (userId == Guid.Empty) return Unauthorized();

        var start = DateTimeOffset.Parse(req.StartDateTime);
        var end = DateTimeOffset.Parse(req.EndDateTime);

        try
        {
            var hold = await holdService.CreateAsync(userId, req.RoomId, start, end);
            return Ok(new { id = hold.Id.ToString(), expiresAt = hold.ExpiresAt.ToString("o") });
        }
        catch (InvalidOperationException ex)
        {
            return Conflict(new { message = ex.Message });
        }
    }

    [HttpDelete("{id:guid}")]
    public async Task<IActionResult> Release(Guid id)
    {
        var userId = CurrentUserId();
        if (userId == Guid.Empty) return Unauthorized();

        await holdService.ReleaseAsync(id, userId);
        return NoContent();
    }
}

public record CreateSlotHoldRequest(Guid RoomId, string StartDateTime, string EndDateTime);
