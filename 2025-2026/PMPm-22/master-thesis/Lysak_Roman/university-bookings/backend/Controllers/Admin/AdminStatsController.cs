using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Models;

namespace UniversityBookings.Controllers.Admin;

[ApiController]
[Route("api/admin/stats")]
[Authorize(Policy = "AdminOnly")]
public class AdminStatsController(AppDbContext db) : ControllerBase
{
    [HttpGet]
    public async Task<ActionResult<DashboardStatsDto>> GetStats()
    {
        var now = DateTime.UtcNow;
        var todayStart = now.Date;
        var monthStart = new DateTime(now.Year, now.Month, 1, 0, 0, 0, DateTimeKind.Utc);

        var allThisMonth = await db.Appointments
            .Where(a => a.CreatedAt >= monthStart)
            .ToListAsync();

        int totalToday = allThisMonth.Count(a => a.CreatedAt.Date == todayStart);
        int totalMonth = allThisMonth.Count;
        int pending = await db.Appointments.CountAsync(a => a.Status == AppointmentStatus.Pending);
        int cancelled = allThisMonth.Count(a => a.Status == AppointmentStatus.Cancelled);
        double cancelRate = totalMonth > 0 ? (double)cancelled / totalMonth * 100 : 0;

        var rawForStats = await db.Appointments
            .Where(a => a.CreatedAt >= monthStart)
            .Select(a => new { a.RoomId, a.Status, a.CreatedAt })
            .ToListAsync();

        var roomIds = rawForStats
            .Where(a => a.Status != AppointmentStatus.Cancelled)
            .Select(a => a.RoomId)
            .Distinct()
            .ToList();

        var roomNames = await db.Rooms
            .Where(r => roomIds.Contains(r.Id))
            .Select(r => new { r.Id, r.Name })
            .ToDictionaryAsync(r => r.Id, r => r.Name);

        var popularRooms = rawForStats
            .Where(a => a.Status != AppointmentStatus.Cancelled)
            .GroupBy(a => a.RoomId)
            .Select(g => new PopularRoomDto(
                g.Key,
                roomNames.GetValueOrDefault(g.Key, "Unknown"),
                g.Count()))
            .OrderByDescending(p => p.Count)
            .Take(5)
            .ToList();

        var bookingsPerDay = rawForStats
            .GroupBy(a => a.CreatedAt.Date)
            .Select(g => new BookingsPerDayDto(
                g.Key.ToString("yyyy-MM-dd"),
                g.Count()))
            .OrderBy(b => b.Date)
            .ToList();

        return Ok(new DashboardStatsDto(
            totalToday, totalMonth, pending, cancelRate,
            popularRooms, bookingsPerDay));
    }
}
