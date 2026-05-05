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
[Route("api/appointments")]
[Authorize]
public class AppointmentsController(
    AppDbContext db,
    AppointmentService appointmentService,
    NotificationService notificationService,
    AppointmentHubService hubService,
    SlotHoldService holdService) : ControllerBase
{
    private Guid CurrentUserId()
    {
        var oid = User.FindFirst("oid")?.Value
               ?? User.FindFirst("http://schemas.microsoft.com/identity/claims/objectidentifier")?.Value;
        var user = db.Users.FirstOrDefault(u => u.AzureObjectId == oid);
        return user?.Id ?? Guid.Empty;
    }

    [HttpGet("my")]
    public async Task<ActionResult<IEnumerable<AppointmentDto>>> GetMy()
    {
        var userId = CurrentUserId();
        var appointments = await db.Appointments
            .Include(a => a.Room)
            .Include(a => a.ClientUser)
            .Where(a => a.ClientUserId == userId)
            .OrderByDescending(a => a.StartDateTime)
            .ToListAsync();

        return Ok(appointments.Select(a => a.ToDto()));
    }

    [HttpPost]
    public async Task<ActionResult<AppointmentDto>> Create([FromBody] CreateAppointmentRequest req)
    {
        var userId = CurrentUserId();
        if (userId == Guid.Empty) return Unauthorized();

        try
        {
            var appointment = await appointmentService.CreateAsync(userId, req);
            var dto = appointment.ToDto();

            // Release the user's hold for this room (they've confirmed the booking)
            await holdService.ReleaseByUserAndRoomAsync(userId, req.RoomId);

            await hubService.NotifyAdminsNewAppointment(dto);

            // Notify responsible staff member for this room (SignalR + email)
            if (appointment.Room?.ResponsiblePersonId != null)
            {
                var staff = await db.StaffMembers
                    .Include(sm => sm.User)
                    .FirstOrDefaultAsync(sm => sm.Id == appointment.Room.ResponsiblePersonId);
                if (staff != null)
                {
                    // Skip SignalR for admins — they already receive via the "admins" group
                    if (staff.User?.Role != UserRole.Admin)
                        await hubService.NotifyUser(staff.UserId.ToString(), "NewAppointmentCreated", dto);
                    if (staff.User?.Email is { } staffEmail)
                        _ = notificationService.SendStaffNewBookingAsync(appointment, staffEmail);
                }
            }

            return CreatedAtAction(nameof(GetMy), dto);
        }
        catch (InvalidOperationException ex)
        {
            return Conflict(new { message = ex.Message });
        }
    }

    [HttpDelete("{id:guid}")]
    public async Task<IActionResult> Cancel(Guid id, [FromBody] CancelAppointmentRequest? req)
    {
        var userId = CurrentUserId();
        var currentUser = await db.Users.FindAsync(userId);
        bool isAdmin = currentUser?.Role == UserRole.Admin;
        try
        {
            await appointmentService.CancelAsync(id, userId, isAdmin, req?.Reason);

            var appointment = await db.Appointments
                .Include(a => a.Room).ThenInclude(r => r.ResponsiblePerson).ThenInclude(sm => sm!.User)
                .Include(a => a.ClientUser)
                .FirstOrDefaultAsync(a => a.Id == id);

            if (appointment != null)
            {
                _ = notificationService.SendCancellationAsync(appointment);
                await hubService.NotifyUser(appointment.ClientUserId.ToString(),
                    "AppointmentCancelled", appointment.ToDto());
                await hubService.NotifyAdminsStatusChanged(appointment.ToDto());

                // Notify responsible staff member only when the client (not admin) cancels
                if (!isAdmin && appointment.Room?.ResponsiblePerson != null)
                {
                    await hubService.NotifyUser(appointment.Room.ResponsiblePerson.UserId.ToString(),
                        "AppointmentCancelled", appointment.ToDto());
                    // Email resolved here (within request scope) before fire-and-forget
                    if (appointment.Room.ResponsiblePerson.User?.Email is { } staffEmail)
                        _ = notificationService.SendStaffCancellationAsync(appointment, staffEmail);
                }
            }

            return NoContent();
        }
        catch (KeyNotFoundException) { return NotFound(); }
        catch (UnauthorizedAccessException) { return Forbid(); }
        catch (InvalidOperationException ex) { return BadRequest(new { message = ex.Message }); }
    }

    [HttpPatch("{id:guid}/status")]
    public async Task<ActionResult<AppointmentDto>> UpdateStatus(
        Guid id, [FromBody] UpdateAppointmentStatusRequest req)
    {
        var userId = CurrentUserId();
        var currentUser = await db.Users.FindAsync(userId);
        if (currentUser?.Role != UserRole.Admin) return Forbid();

        try
        {
            var appointment = await appointmentService.UpdateStatusAsync(id, req.Status);
            var dto = appointment.ToDto();

            await hubService.NotifyAdminsStatusChanged(dto);
            await hubService.NotifyUser(appointment.ClientUserId.ToString(),
                req.Status == AppointmentStatus.Confirmed ? "AppointmentConfirmed" : "AppointmentCancelled",
                dto);

            if (req.Status == AppointmentStatus.Confirmed)
                _ = notificationService.SendConfirmationAsync(appointment);
            else if (req.Status == AppointmentStatus.Cancelled)
                _ = notificationService.SendCancellationAsync(appointment);

            return Ok(dto);
        }
        catch (KeyNotFoundException) { return NotFound(); }
    }
}
