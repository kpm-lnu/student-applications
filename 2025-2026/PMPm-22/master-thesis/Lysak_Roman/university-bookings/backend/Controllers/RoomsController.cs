using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Models;
using UniversityBookings.Services;

namespace UniversityBookings.Controllers;

[ApiController]
[Route("api/rooms")]
public class RoomsController(AppDbContext db, AvailabilityService availabilityService) : ControllerBase
{
    [HttpGet]
    public async Task<ActionResult<IEnumerable<RoomDto>>> GetAll([FromQuery] string? type)
    {
        var query = db.Rooms
            .Include(r => r.RoomType)
            .Include(r => r.Address)
            .Include(r => r.ResponsiblePerson).ThenInclude(sm => sm!.User)
            .Include(r => r.Availabilities)
            .Where(r => r.IsActive)
            .AsQueryable();

        if (!string.IsNullOrWhiteSpace(type))
            query = query.Where(r => r.RoomType != null && r.RoomType.Name == type.ToLowerInvariant());

        var rooms = await query
            .OrderBy(r => r.RoomType != null ? r.RoomType.Name : string.Empty).ThenBy(r => r.Name)
            .ToListAsync();

        return Ok(rooms.Select(r => r.ToDto()));
    }

    [HttpGet("{id:guid}")]
    public async Task<ActionResult<RoomDto>> GetById(Guid id)
    {
        var room = await db.Rooms
            .Include(r => r.RoomType)
            .Include(r => r.Address)
            .Include(r => r.ResponsiblePerson).ThenInclude(sm => sm!.User)
            .Include(r => r.Availabilities)
            .FirstOrDefaultAsync(r => r.Id == id);

        if (room is null) return NotFound();
        return Ok(room.ToDto());
    }

    [HttpGet("{id:guid}/available-slots")]
    public async Task<ActionResult<IEnumerable<TimeSlotDto>>> GetAvailableSlots(
        Guid id, [FromQuery] string date, [FromQuery] int duration = 60)
    {
        if (string.IsNullOrWhiteSpace(date)) return BadRequest("date query param is required.");

        var slotMode = await db.Rooms
            .Where(r => r.Id == id && r.IsActive)
            .Select(r => (SlotMode?)r.SlotMode)
            .FirstOrDefaultAsync();

        if (slotMode != SlotMode.Para)
        {
            int[] validDurations = [40, 60, 80, 120];
            if (!validDurations.Contains(duration))
                return BadRequest("duration must be one of: 40, 60, 80, 120.");
        }

        Guid? currentUserId = null;
        if (User.Identity?.IsAuthenticated == true)
        {
            var oid = User.FindFirst("oid")?.Value
                   ?? User.FindFirst("http://schemas.microsoft.com/identity/claims/objectidentifier")?.Value;
            if (oid is not null)
                currentUserId = (await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == oid))?.Id;
        }

        var slots = await availabilityService.GetAvailableSlotsAsync(id, date, duration, currentUserId);
        return Ok(slots);
    }
}
