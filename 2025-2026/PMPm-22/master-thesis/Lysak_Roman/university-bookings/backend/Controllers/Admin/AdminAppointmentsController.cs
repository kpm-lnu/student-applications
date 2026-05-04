using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Models;

namespace UniversityBookings.Controllers.Admin;

[ApiController]
[Route("api/admin/appointments")]
[Authorize(Policy = "AdminOnly")]
public class AdminAppointmentsController(AppDbContext db) : ControllerBase
{
    [HttpGet]
    public async Task<ActionResult<IEnumerable<AppointmentDto>>> GetAll(
        [FromQuery] string? status,
        [FromQuery] string? from,
        [FromQuery] string? to,
        [FromQuery] Guid? roomId)
    {
        var query = db.Appointments
            .Include(a => a.Room)
            .Include(a => a.ClientUser)
            .AsQueryable();

        if (Enum.TryParse<AppointmentStatus>(status, out var parsedStatus))
            query = query.Where(a => a.Status == parsedStatus);

        if (DateTimeOffset.TryParse(from, out var fromDate))
            query = query.Where(a => a.StartDateTime >= fromDate);

        if (DateTimeOffset.TryParse(to, out var toDate))
            query = query.Where(a => a.StartDateTime <= toDate);

        if (roomId.HasValue)
            query = query.Where(a => a.RoomId == roomId.Value);

        var results = await query
            .OrderByDescending(a => a.StartDateTime)
            .ToListAsync();

        return Ok(results.Select(a => a.ToDto()));
    }
}
