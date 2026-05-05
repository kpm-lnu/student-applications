using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Models;

namespace UniversityBookings.Controllers.Admin;

[ApiController]
[Route("api/admin/rooms")]
[Authorize(Policy = "AdminOnly")]
public class AdminRoomsController(AppDbContext db) : ControllerBase
{
    // ── Rooms CRUD ────────────────────────────────────────────────────────────

    [HttpGet]
    public async Task<ActionResult<IEnumerable<RoomDto>>> GetAll()
    {
        var rooms = await db.Rooms
            .Include(r => r.RoomType)
            .Include(r => r.Address)
            .Include(r => r.ResponsiblePerson).ThenInclude(sm => sm!.User)
            .Include(r => r.Availabilities)
            .OrderBy(r => r.Name)
            .ToListAsync();
        return Ok(rooms.Select(r => r.ToDto()));
    }

    [HttpGet("{id:guid}")]
    public async Task<ActionResult<RoomDto>> GetById(Guid id)
    {
        var room = await LoadRoom(id);
        if (room is null) return NotFound();
        return Ok(room.ToDto());
    }

    [HttpPost]
    public async Task<ActionResult<RoomDto>> Create([FromBody] CreateRoomRequest req)
    {
        var room = new Room
        {
            Name = req.Name,
            RoomNumber = req.RoomNumber,
            RoomTypeId = req.RoomTypeId,
            AddressId = req.AddressId,
            Description = req.Description,
            Capacity = req.Capacity,
            IsActive = req.IsActive,
            ResponsiblePersonId = req.ResponsiblePersonId,
            SlotMode = req.SlotMode,
        };
        db.Rooms.Add(room);
        await db.SaveChangesAsync();

        return CreatedAtAction(nameof(GetById), new { id = room.Id },
            (await LoadRoom(room.Id))!.ToDto());
    }

    [HttpPut("{id:guid}")]
    public async Task<ActionResult<RoomDto>> Update(Guid id, [FromBody] UpdateRoomRequest req)
    {
        var room = await db.Rooms.FindAsync(id);
        if (room is null) return NotFound();

        room.Name = req.Name;
        room.RoomNumber = req.RoomNumber;
        room.RoomTypeId = req.RoomTypeId;
        room.AddressId = req.AddressId;
        room.Description = req.Description;
        room.Capacity = req.Capacity;
        room.IsActive = req.IsActive;
        room.ResponsiblePersonId = req.ResponsiblePersonId;
        room.SlotMode = req.SlotMode;

        await db.SaveChangesAsync();
        return Ok((await LoadRoom(id))!.ToDto());
    }

    [HttpDelete("{id:guid}")]
    public async Task<IActionResult> Delete(Guid id)
    {
        var room = await db.Rooms.FindAsync(id);
        if (room is null) return NotFound();

        var hasActiveAppointments = await db.Appointments.AnyAsync(a =>
            a.RoomId == id &&
            (a.Status == AppointmentStatus.Pending || a.Status == AppointmentStatus.Confirmed));

        if (hasActiveAppointments)
            return Conflict("Неможливо видалити приміщення: існують активні або підтверджені бронювання.");

        var closedAppointments = await db.Appointments
            .Where(a => a.RoomId == id &&
                (a.Status == AppointmentStatus.Cancelled || a.Status == AppointmentStatus.Completed))
            .ToListAsync();

        db.Appointments.RemoveRange(closedAppointments);
        db.Rooms.Remove(room);
        await db.SaveChangesAsync();
        return NoContent();
    }

    // ── Availability ──────────────────────────────────────────────────────────

    [HttpPost("{id:guid}/availability")]
    public async Task<ActionResult<AvailabilityDto>> AddAvailability(
        Guid id, [FromBody] SetAvailabilityRequest req)
    {
        if (!await db.Rooms.AnyAsync(r => r.Id == id)) return NotFound();

        if (!TimeSpan.TryParse(req.StartTime, out var start) ||
            !TimeSpan.TryParse(req.EndTime, out var end))
            return BadRequest("StartTime and EndTime must be in HH:mm format.");

        if (end <= start)
            return BadRequest("EndTime must be after StartTime.");

        // Remove existing entry for that day if present (replace semantics)
        var existing = await db.Availabilities
            .FirstOrDefaultAsync(av => av.RoomId == id && av.DayOfWeek == req.DayOfWeek);
        if (existing is not null)
            db.Availabilities.Remove(existing);

        var av = new Availability
        {
            RoomId = id,
            DayOfWeek = req.DayOfWeek,
            StartTime = start,
            EndTime = end,
            AvailableParaIndices = req.AvailableParaIndices ?? [],
        };
        db.Availabilities.Add(av);
        await db.SaveChangesAsync();

        return CreatedAtAction(nameof(GetById), new { id }, av.ToDto());
    }

    [HttpDelete("{id:guid}/availability/{availId:guid}")]
    public async Task<IActionResult> DeleteAvailability(Guid id, Guid availId)
    {
        var av = await db.Availabilities
            .FirstOrDefaultAsync(av => av.Id == availId && av.RoomId == id);
        if (av is null) return NotFound();

        db.Availabilities.Remove(av);
        await db.SaveChangesAsync();
        return NoContent();
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private async Task<Room?> LoadRoom(Guid id) =>
        await db.Rooms
            .Include(r => r.RoomType)
            .Include(r => r.Address)
            .Include(r => r.ResponsiblePerson).ThenInclude(sm => sm!.User)
            .Include(r => r.Availabilities)
            .FirstOrDefaultAsync(r => r.Id == id);
}
