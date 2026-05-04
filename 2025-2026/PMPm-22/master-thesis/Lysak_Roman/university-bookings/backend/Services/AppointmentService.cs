using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Models;

namespace UniversityBookings.Services;

public class AppointmentService(AppDbContext db)
{
    private static readonly TimeZoneInfo Kyiv = TimeZoneInfo.FindSystemTimeZoneById("FLE Standard Time");

    public async Task<Appointment> CreateAsync(Guid clientUserId, CreateAppointmentRequest req)
    {
        var room = await db.Rooms
            .Include(r => r.Availabilities)
            .FirstOrDefaultAsync(r => r.Id == req.RoomId && r.IsActive)
            ?? throw new InvalidOperationException("Кімната не знайдена або недоступна.");

        var start = DateTimeOffset.Parse(req.StartDateTime).ToUniversalTime();
        var localStart = TimeZoneInfo.ConvertTimeFromUtc(start.UtcDateTime, Kyiv);
        var availability = room.Availabilities.FirstOrDefault(av => av.DayOfWeek == (int)localStart.DayOfWeek);

        if (availability is null)
            throw new InvalidOperationException("Кімната не працює в цей день тижня.");

        DateTimeOffset end;
        int durationMinutes;

        if (room.SlotMode == SlotMode.Para)
        {
            var localStartTime = localStart.TimeOfDay;
            var matched = UniversityPara.Schedule.FirstOrDefault(p => p.Start == localStartTime);
            if (matched.Index == 0)
                throw new InvalidOperationException("Невалідний часовий слот для режиму пар.");

            if (availability.AvailableParaIndices.Count > 0 &&
                !availability.AvailableParaIndices.Contains(matched.Index))
                throw new InvalidOperationException("Ця пара недоступна для бронювання в цей день.");

            var localOffset = Kyiv.GetUtcOffset(localStart);
            end = new DateTimeOffset(localStart.Year, localStart.Month, localStart.Day,
                matched.End.Hours, matched.End.Minutes, 0, localOffset).ToUniversalTime();
            durationMinutes = 80;
        }
        else
        {
            int[] validDurations = [40, 60, 80, 120];
            if (!validDurations.Contains(req.DurationMinutes))
                throw new InvalidOperationException("Тривалість має бути 40, 60, 80 або 120 хвилин.");

            end = start.AddMinutes(req.DurationMinutes);
            durationMinutes = req.DurationMinutes;

            var bookingStartTime = localStart.TimeOfDay;
            var bookingEndTime = bookingStartTime.Add(TimeSpan.FromMinutes(req.DurationMinutes));

            if (bookingStartTime < availability.StartTime || bookingEndTime > availability.EndTime)
                throw new InvalidOperationException(
                    $"Час бронювання виходить за межі робочих годин кімнати ({availability.StartTime:hh\\:mm}–{availability.EndTime:hh\\:mm}).");
        }

        bool conflict = await db.Appointments.AnyAsync(a =>
            a.RoomId == req.RoomId &&
            a.Status != AppointmentStatus.Cancelled &&
            a.StartDateTime < end &&
            a.EndDateTime > start);

        if (conflict)
            throw new InvalidOperationException("Обраний час вже зайнятий. Будь ласка, оберіть інший.");

        bool holdConflict = await db.SlotHolds.AnyAsync(h =>
            h.RoomId == req.RoomId &&
            h.UserId != clientUserId &&
            h.ExpiresAt > DateTimeOffset.UtcNow &&
            h.StartDateTime < end &&
            h.EndDateTime > start);

        if (holdConflict)
            throw new InvalidOperationException("Цей час тимчасово заблоковано іншим користувачем. Будь ласка, зачекайте або оберіть інший час.");

        var appointment = new Appointment
        {
            RoomId = req.RoomId,
            ClientUserId = clientUserId,
            DurationMinutes = durationMinutes,
            StartDateTime = start,
            EndDateTime = end,
            Status = AppointmentStatus.Pending,
            Notes = req.Notes,
        };

        db.Appointments.Add(appointment);
        await db.SaveChangesAsync();

        return await db.Appointments
            .Include(a => a.Room)
            .Include(a => a.ClientUser)
            .FirstAsync(a => a.Id == appointment.Id);
    }

    public async Task CancelAsync(Guid appointmentId, Guid requestingUserId, bool isAdmin, string? reason)
    {
        var appointment = await db.Appointments
            .FirstOrDefaultAsync(a => a.Id == appointmentId)
            ?? throw new KeyNotFoundException("Запис не знайдено.");

        if (!isAdmin && appointment.ClientUserId != requestingUserId)
            throw new UnauthorizedAccessException("Ви не маєте права скасувати цей запис.");

        if (appointment.Status == AppointmentStatus.Cancelled)
            throw new InvalidOperationException("Запис вже скасовано.");

        if (appointment.Status == AppointmentStatus.Completed)
            throw new InvalidOperationException("Не можна скасувати завершений запис.");

        appointment.Status = AppointmentStatus.Cancelled;
        appointment.CancelledAt = DateTime.UtcNow;
        appointment.CancellationReason = reason;
        await db.SaveChangesAsync();
    }

    public async Task<Appointment> UpdateStatusAsync(Guid appointmentId, AppointmentStatus status)
    {
        var appointment = await db.Appointments
            .Include(a => a.Room)
            .Include(a => a.ClientUser)
            .FirstOrDefaultAsync(a => a.Id == appointmentId)
            ?? throw new KeyNotFoundException("Запис не знайдено.");

        appointment.Status = status;
        if (status == AppointmentStatus.Cancelled)
            appointment.CancelledAt = DateTime.UtcNow;

        await db.SaveChangesAsync();
        return appointment;
    }
}
