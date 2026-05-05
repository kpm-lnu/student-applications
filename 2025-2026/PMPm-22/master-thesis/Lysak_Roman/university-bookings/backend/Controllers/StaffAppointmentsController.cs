using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Hubs;
using UniversityBookings.Models;
using UniversityBookings.Services;

namespace UniversityBookings.Controllers;

[ApiController]
[Route("api/staff/appointments")]
[Authorize]
public class StaffAppointmentsController(
    AppDbContext db,
    AppointmentService appointmentService,
    NotificationService notificationService,
    AppointmentHubService hubService) : ControllerBase
{
    private async Task<(User? user, StaffMember? staffMember)> GetCurrentStaff()
    {
        var oid = User.FindFirst("oid")?.Value
               ?? User.FindFirst("http://schemas.microsoft.com/identity/claims/objectidentifier")?.Value;
        var user = await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == oid);
        if (user is null || user.Role != UserRole.Staff) return (null, null);

        var staffMember = await db.StaffMembers.FirstOrDefaultAsync(sm => sm.UserId == user.Id);
        return (user, staffMember);
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<AppointmentDto>>> GetMyRoomsAppointments(
        [FromQuery] string? status)
    {
        var (user, staffMember) = await GetCurrentStaff();
        if (user is null) return Forbid();
        if (staffMember is null) return Ok(Array.Empty<AppointmentDto>());

        var roomIds = await db.Rooms
            .Where(r => r.ResponsiblePersonId == staffMember.Id)
            .Select(r => r.Id)
            .ToListAsync();

        var query = db.Appointments
            .Include(a => a.Room).ThenInclude(r => r!.RoomType)
            .Include(a => a.ClientUser)
            .Where(a => roomIds.Contains(a.RoomId));

        if (!string.IsNullOrWhiteSpace(status) && Enum.TryParse<AppointmentStatus>(status, out var parsed))
            query = query.Where(a => a.Status == parsed);

        var appointments = await query
            .OrderByDescending(a => a.StartDateTime)
            .ToListAsync();

        return Ok(appointments.Select(a => a.ToDto()));
    }

    [HttpPatch("{id:guid}/status")]
    public async Task<ActionResult<AppointmentDto>> UpdateStatus(
        Guid id, [FromBody] UpdateAppointmentStatusRequest req)
    {
        var (user, staffMember) = await GetCurrentStaff();
        if (user is null || staffMember is null) return Forbid();

        // Only Confirmed and Cancelled are allowed for staff
        if (req.Status != AppointmentStatus.Confirmed && req.Status != AppointmentStatus.Cancelled)
            return BadRequest(new { message = "Персонал може тільки підтверджувати або скасовувати записи." });

        var appointment = await db.Appointments
            .Include(a => a.Room)
            .FirstOrDefaultAsync(a => a.Id == id);

        if (appointment is null) return NotFound();
        if (appointment.Room?.ResponsiblePersonId != staffMember.Id) return Forbid();

        try
        {
            var updated = await appointmentService.UpdateStatusAsync(id, req.Status);
            var dto = updated.ToDto();

            await hubService.NotifyAdminsStatusChanged(dto);
            await hubService.NotifyUser(updated.ClientUserId.ToString(),
                req.Status == AppointmentStatus.Confirmed ? "AppointmentConfirmed" : "AppointmentCancelled",
                dto);

            if (req.Status == AppointmentStatus.Confirmed)
                _ = notificationService.SendConfirmationAsync(updated);
            else if (req.Status == AppointmentStatus.Cancelled)
                _ = notificationService.SendCancellationAsync(updated);

            return Ok(dto);
        }
        catch (KeyNotFoundException) { return NotFound(); }
    }
}
