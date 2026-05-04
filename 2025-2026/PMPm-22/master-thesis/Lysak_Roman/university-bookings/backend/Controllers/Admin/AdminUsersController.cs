using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Models;

namespace UniversityBookings.Controllers.Admin;

[ApiController]
[Route("api/admin/users")]
[Authorize(Policy = "AdminOnly")]
public class AdminUsersController(AppDbContext db) : ControllerBase
{
    [HttpGet]
    public async Task<ActionResult<IEnumerable<UserDto>>> GetAll()
    {
        var users = await db.Users
            .OrderBy(u => u.DisplayName)
            .ToListAsync();
        return Ok(users.Select(u => u.ToDto()));
    }

    private async Task<Guid?> GetCurrentUserIdAsync()
    {
        var oid = User.FindFirst("oid")?.Value
               ?? User.FindFirst("http://schemas.microsoft.com/identity/claims/objectidentifier")?.Value;
        if (oid is null) return null;
        var u = await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == oid);
        return u?.Id;
    }

    [HttpPut("{id:guid}/role")]
    public async Task<IActionResult> UpdateRole(Guid id, [FromBody] UpdateRoleRequest req)
    {
        var currentId = await GetCurrentUserIdAsync();
        if (currentId == id)
            return StatusCode(403, "Неможливо змінити власну роль.");

        var user = await db.Users.FindAsync(id);
        if (user is null) return NotFound();

        if (user.Role == UserRole.Staff && req.Role != UserRole.Staff)
        {
            var staffMember = await db.StaffMembers.FirstOrDefaultAsync(sm => sm.UserId == id);
            if (staffMember != null)
            {
                var hasRooms = await db.Rooms.AnyAsync(r => r.ResponsiblePersonId == staffMember.Id);
                if (hasRooms)
                    return Conflict("Неможливо змінити роль: зніміть відповідальність за кімнати перед зміною ролі.");

                db.StaffMembers.Remove(staffMember);
            }
        }

        user.Role = req.Role;

        if (req.Role == UserRole.Staff)
        {
            var exists = await db.StaffMembers.AnyAsync(sm => sm.UserId == id);
            if (!exists)
                db.StaffMembers.Add(new StaffMember { UserId = id });
        }

        await db.SaveChangesAsync();
        return Ok(user.ToDto());
    }

    [HttpDelete("{id:guid}")]
    public async Task<IActionResult> DeleteUser(Guid id)
    {
        var currentId = await GetCurrentUserIdAsync();
        if (currentId == id)
            return StatusCode(403, "Неможливо видалити власний акаунт.");

        var user = await db.Users.FindAsync(id);
        if (user is null) return NotFound();

        var hasActiveAppointments = await db.Appointments.AnyAsync(a =>
            a.ClientUserId == id &&
            (a.Status == AppointmentStatus.Pending || a.Status == AppointmentStatus.Confirmed));

        if (hasActiveAppointments)
            return Conflict("Неможливо видалити користувача: існують активні або підтверджені бронювання.");

        var closedAppointments = await db.Appointments
            .Where(a => a.ClientUserId == id &&
                (a.Status == AppointmentStatus.Cancelled || a.Status == AppointmentStatus.Completed))
            .ToListAsync();

        db.Appointments.RemoveRange(closedAppointments);
        db.Users.Remove(user);
        await db.SaveChangesAsync();
        return NoContent();
    }

    [HttpDelete("bulk-students")]
    public async Task<IActionResult> DeleteBulkStudents([FromQuery] DateTime before)
    {
        var beforeUtc = DateTime.SpecifyKind(before.Date.AddDays(1).AddTicks(-1), DateTimeKind.Utc);

        var students = await db.Users
            .Where(u => u.Role == UserRole.Student && u.CreatedAt <= beforeUtc)
            .ToListAsync();

        if (students.Count == 0)
            return Ok(new { deleted = 0, skipped = 0 });

        var studentIds = students.Select(u => u.Id).ToList();

        var activeUserIds = await db.Appointments
            .Where(a => studentIds.Contains(a.ClientUserId) &&
                (a.Status == AppointmentStatus.Pending || a.Status == AppointmentStatus.Confirmed))
            .Select(a => a.ClientUserId)
            .Distinct()
            .ToListAsync();

        var deletableStudents = students.Where(u => !activeUserIds.Contains(u.Id)).ToList();

        if (deletableStudents.Count == 0)
            return Ok(new { deleted = 0, skipped = students.Count });

        var deletableIds = deletableStudents.Select(u => u.Id).ToList();
        var closedAppointments = await db.Appointments
            .Where(a => deletableIds.Contains(a.ClientUserId) &&
                (a.Status == AppointmentStatus.Cancelled || a.Status == AppointmentStatus.Completed))
            .ToListAsync();

        db.Appointments.RemoveRange(closedAppointments);
        db.Users.RemoveRange(deletableStudents);
        await db.SaveChangesAsync();

        return Ok(new { deleted = deletableStudents.Count, skipped = activeUserIds.Count });
    }
}
