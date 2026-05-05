using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Models;

namespace UniversityBookings.Services;

public class AvailabilityService(AppDbContext db)
{
    private static readonly TimeZoneInfo Kyiv = TimeZoneInfo.FindSystemTimeZoneById("FLE Standard Time");

    public async Task<IEnumerable<TimeSlotDto>> GetAvailableSlotsAsync(
        Guid roomId, string dateString, int durationMinutes, Guid? excludeUserId = null)
    {
        var room = await db.Rooms
            .Include(r => r.Availabilities)
            .FirstOrDefaultAsync(r => r.Id == roomId && r.IsActive);

        if (room is null) return [];

        if (!DateOnly.TryParse(dateString, out var date)) return [];

        int dayOfWeek = (int)date.DayOfWeek;

        var availability = room.Availabilities.FirstOrDefault(av => av.DayOfWeek == dayOfWeek);
        if (availability is null) return [];

        var localOffset = Kyiv.GetUtcOffset(new DateTime(date.Year, date.Month, date.Day));
        var dayStartLocal = new DateTimeOffset(date.Year, date.Month, date.Day, 0, 0, 0, localOffset);
        var dayEndLocal = dayStartLocal.AddDays(1);

        var bookedSlots = await db.Appointments
            .Where(a =>
                a.RoomId == roomId &&
                a.StartDateTime >= dayStartLocal &&
                a.StartDateTime < dayEndLocal &&
                a.Status != AppointmentStatus.Cancelled)
            .Select(a => new { a.StartDateTime, a.EndDateTime })
            .ToListAsync();

        var activeHolds = await db.SlotHolds
            .Where(h => h.RoomId == roomId
                     && h.ExpiresAt > DateTimeOffset.UtcNow
                     && (excludeUserId == null || h.UserId != excludeUserId))
            .Select(h => new { h.StartDateTime, h.EndDateTime })
            .ToListAsync();

        if (room.SlotMode == SlotMode.Para)
        {
            var paras = availability.AvailableParaIndices.Count > 0
                ? UniversityPara.Schedule.Where(p => availability.AvailableParaIndices.Contains(p.Index)).ToArray()
                : UniversityPara.Schedule;

            var paraSlots = new List<TimeSlotDto>();
            foreach (var para in paras)
            {
                var slotStart = new DateTimeOffset(date.Year, date.Month, date.Day,
                    para.Start.Hours, para.Start.Minutes, 0, localOffset);
                var slotEnd = new DateTimeOffset(date.Year, date.Month, date.Day,
                    para.End.Hours, para.End.Minutes, 0, localOffset);

                bool isBooked = bookedSlots.Any(b =>
                    b.StartDateTime < slotEnd && b.EndDateTime > slotStart);

                bool isHeld = !isBooked && activeHolds.Any(h =>
                    h.StartDateTime < slotEnd && h.EndDateTime > slotStart);

                paraSlots.Add(new TimeSlotDto(
                    slotStart.ToString("o"),
                    slotEnd.ToString("o"),
                    !isBooked && !isHeld,
                    isHeld
                ));
            }
            return paraSlots;
        }

        // Interval mode — existing logic unchanged
        var openHour = availability.StartTime.Minutes == 0
            ? availability.StartTime.Hours
            : availability.StartTime.Hours + 1;
        var slotStart2 = TimeSpan.FromHours(openHour);
        var slotDuration = TimeSpan.FromMinutes(durationMinutes);
        var workEnd = availability.EndTime;

        var slots = new List<TimeSlotDto>();

        while (slotStart2 + slotDuration <= workEnd)
        {
            var slotStartDto = new DateTimeOffset(date.Year, date.Month, date.Day,
                slotStart2.Hours, slotStart2.Minutes, 0, localOffset);
            var slotEndDto = slotStartDto.Add(slotDuration);

            bool isBooked = bookedSlots.Any(b =>
                b.StartDateTime < slotEndDto && b.EndDateTime > slotStartDto);

            bool isHeld = !isBooked && activeHolds.Any(h =>
                h.StartDateTime < slotEndDto && h.EndDateTime > slotStartDto);

            slots.Add(new TimeSlotDto(
                slotStartDto.ToString("o"),
                slotEndDto.ToString("o"),
                !isBooked && !isHeld,
                isHeld
            ));

            slotStart2 = slotStart2.Add(TimeSpan.FromHours(1));
        }

        return slots;
    }
}
