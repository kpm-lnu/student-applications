using System.Data;
using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.Hubs;
using UniversityBookings.Models;

namespace UniversityBookings.Services;

public class SlotHoldService(AppDbContext db, IHubContext<AppointmentHub> hub)
{
    private static readonly TimeSpan HoldDuration = TimeSpan.FromMinutes(5);
    private static readonly TimeZoneInfo Kyiv = TimeZoneInfo.FindSystemTimeZoneById("FLE Standard Time");

    public async Task<SlotHold> CreateAsync(Guid userId, Guid roomId, DateTimeOffset start, DateTimeOffset end)
    {
        await CleanupExpiredAsync();

        var room = await db.Rooms.FindAsync(roomId);
        if (room?.SlotMode == SlotMode.Para)
        {
            var localStart = TimeZoneInfo.ConvertTimeFromUtc(start.UtcDateTime, Kyiv);
            var localEnd = TimeZoneInfo.ConvertTimeFromUtc(end.UtcDateTime, Kyiv);
            if (!UniversityPara.IsValidPara(localStart.TimeOfDay, localEnd.TimeOfDay))
                throw new InvalidOperationException("Невалідний часовий слот для режиму пар.");
        }

        SlotHold hold;
        List<SlotHold> released = [];

        await using var tx = await db.Database.BeginTransactionAsync(IsolationLevel.Serializable);

        bool apptConflict = await db.Appointments.AnyAsync(a =>
            a.RoomId == roomId &&
            a.Status != AppointmentStatus.Cancelled &&
            a.StartDateTime < end &&
            a.EndDateTime > start);

        if (apptConflict)
        {
            await tx.RollbackAsync();
            throw new InvalidOperationException("Обраний час вже зайнятий.");
        }

        bool holdConflict = await db.SlotHolds.AnyAsync(h =>
            h.RoomId == roomId &&
            h.UserId != userId &&
            h.ExpiresAt > DateTimeOffset.UtcNow &&
            h.StartDateTime < end &&
            h.EndDateTime > start);

        if (holdConflict)
        {
            await tx.RollbackAsync();
            throw new InvalidOperationException("Цей час тимчасово заблоковано іншим користувачем.");
        }

        // Release previous holds by this user for this room
        var existing = await db.SlotHolds
            .Where(h => h.RoomId == roomId && h.UserId == userId)
            .ToListAsync();

        if (existing.Count > 0)
        {
            db.SlotHolds.RemoveRange(existing);
            released.AddRange(existing);
        }

        hold = new SlotHold
        {
            RoomId = roomId,
            UserId = userId,
            StartDateTime = start,
            EndDateTime = end,
            ExpiresAt = DateTimeOffset.UtcNow.Add(HoldDuration),
        };

        db.SlotHolds.Add(hold);
        await db.SaveChangesAsync();
        await tx.CommitAsync();

        foreach (var old in released)
            await BroadcastReleasedAsync(old);

        await hub.Clients.All.SendAsync("SlotHoldCreated", new
        {
            roomId = roomId.ToString(),
            startDateTime = start.ToString("o"),
            endDateTime = end.ToString("o"),
            expiresAt = hold.ExpiresAt.ToString("o"),
        });

        return hold;
    }

    public async Task ReleaseAsync(Guid holdId, Guid userId)
    {
        var hold = await db.SlotHolds.FindAsync(holdId);
        if (hold == null || hold.UserId != userId) return;

        db.SlotHolds.Remove(hold);
        await db.SaveChangesAsync();
        await BroadcastReleasedAsync(hold);
    }

    public async Task ReleaseByUserAndRoomAsync(Guid userId, Guid roomId)
    {
        var holds = await db.SlotHolds
            .Where(h => h.UserId == userId && h.RoomId == roomId)
            .ToListAsync();

        if (holds.Count == 0) return;

        db.SlotHolds.RemoveRange(holds);
        await db.SaveChangesAsync();

        foreach (var h in holds)
            await BroadcastReleasedAsync(h);
    }

    public async Task CleanupExpiredAsync()
    {
        var expired = await db.SlotHolds
            .Where(h => h.ExpiresAt <= DateTimeOffset.UtcNow)
            .ToListAsync();

        if (expired.Count == 0) return;

        db.SlotHolds.RemoveRange(expired);
        await db.SaveChangesAsync();

        foreach (var h in expired)
            await BroadcastReleasedAsync(h);
    }

    private Task BroadcastReleasedAsync(SlotHold hold) =>
        hub.Clients.All.SendAsync("SlotHoldReleased", new
        {
            roomId = hold.RoomId.ToString(),
            startDateTime = hold.StartDateTime.ToString("o"),
            endDateTime = hold.EndDateTime.ToString("o"),
        });
}
