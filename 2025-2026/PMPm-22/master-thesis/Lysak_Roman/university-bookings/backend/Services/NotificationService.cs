using Azure.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.Graph;
using Microsoft.Graph.Models;
using UniversityBookings.Data;
using UniversityBookings.Models;

namespace UniversityBookings.Services;

public class NotificationService(IServiceScopeFactory scopeFactory, IConfiguration config, ILogger<NotificationService> logger)
{
    private GraphServiceClient BuildGraphClient()
    {
        var tenantId = config["Graph:TenantId"]!;
        var clientId = config["Graph:ClientId"]!;
        var clientSecret = config["Graph:ClientSecret"]!;

        var credential = new ClientSecretCredential(tenantId, clientId, clientSecret);
        return new GraphServiceClient(credential,
            ["https://graph.microsoft.com/.default"]);
    }

    public async Task SendConfirmationAsync(Appointment appointment)
    {
        using var scope = scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        appointment = await db.Appointments
            .Include(a => a.ClientUser)
            .Include(a => a.Room)
                .ThenInclude(r => r.ResponsiblePerson)
                    .ThenInclude(s => s!.User)
            .FirstAsync(a => a.Id == appointment.Id);

        await SendEmailAsync(appointment, isConfirmation: true);

        var eventId = await AddCalendarEventAsync(appointment, db);
        if (eventId != null)
        {
            var tracked = await db.Appointments.FindAsync(appointment.Id);
            if (tracked != null)
            {
                tracked.OutlookEventId = eventId;
                await db.SaveChangesAsync();
            }
        }

        await LogNotificationAsync(db, appointment.ClientUserId, appointment.Id, NotificationType.Email, true);
    }

    public async Task SendCancellationAsync(Appointment appointment)
    {
        using var scope = scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        appointment = await db.Appointments
            .Include(a => a.ClientUser)
            .Include(a => a.Room)
                .ThenInclude(r => r.ResponsiblePerson)
                    .ThenInclude(s => s!.User)
            .FirstAsync(a => a.Id == appointment.Id);

        await SendEmailAsync(appointment, isConfirmation: false);
        await RemoveCalendarEventAsync(appointment);
        await LogNotificationAsync(db, appointment.ClientUserId, appointment.Id, NotificationType.Email, true);
    }

    // Notifies responsible staff that a new booking was created for their room.
    // staffEmail must be resolved by the caller (before fire-and-forget) to avoid disposed DbContext.
    public async Task SendStaffNewBookingAsync(Appointment appointment, string staffEmail)
    {
        var startLocal = appointment.StartDateTime.ToLocalTime();
        var subject = $"Нове бронювання: {appointment.Room.Name}";
        var body = $"""
            <p>Нове бронювання вашого приміщення <strong>{appointment.Room.Name}</strong>.</p>
            <ul>
              <li><strong>Клієнт:</strong> {appointment.ClientUser.DisplayName} ({appointment.ClientUser.Email})</li>
              <li><strong>Дата:</strong> {startLocal:dd MMMM yyyy, HH:mm}</li>
              <li><strong>Тривалість:</strong> {appointment.DurationMinutes} хв</li>
              {(appointment.Notes != null ? $"<li><strong>Примітка:</strong> {appointment.Notes}</li>" : "")}
            </ul>
            """;

        await SendRawEmailAsync(staffEmail, subject, body);
    }

    // Notifies responsible staff that the client cancelled their booking.
    // staffEmail must be resolved by the caller (before fire-and-forget) to avoid disposed DbContext.
    public async Task SendStaffCancellationAsync(Appointment appointment, string staffEmail)
    {
        var startLocal = appointment.StartDateTime.ToLocalTime();
        var subject = $"Клієнт скасував бронювання: {appointment.Room?.Name}";
        var body = $"""
            <p>Клієнт скасував бронювання приміщення <strong>{appointment.Room?.Name}</strong>.</p>
            <ul>
              <li><strong>Клієнт:</strong> {appointment.ClientUser.DisplayName} ({appointment.ClientUser.Email})</li>
              <li><strong>Дата:</strong> {startLocal:dd MMMM yyyy, HH:mm}</li>
              {(appointment.CancellationReason != null ? $"<li><strong>Причина:</strong> {appointment.CancellationReason}</li>" : "")}
            </ul>
            """;

        await SendRawEmailAsync(staffEmail, subject, body);
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    private async Task SendRawEmailAsync(string toEmail, string subject, string htmlBody)
    {
        try
        {
            var graph = BuildGraphClient();
            var message = new Message
            {
                Subject = subject,
                Body = new ItemBody { ContentType = BodyType.Html, Content = htmlBody },
                ToRecipients = [new Recipient { EmailAddress = new EmailAddress { Address = toEmail } }],
            };
            await graph.Users[config["Graph:SenderEmail"] ?? toEmail]
                .SendMail.PostAsync(new Microsoft.Graph.Users.Item.SendMail.SendMailPostRequestBody
                {
                    Message = message,
                    SaveToSentItems = false,
                });
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Failed to send email to {Email}", toEmail);
        }
    }

    private async Task SendEmailAsync(Appointment appointment, bool isConfirmation)
    {
        var recipientEmail = appointment.ClientUser.Email;
        var roomName = appointment.Room.Name;
        var responsibleName = appointment.Room.ResponsiblePerson?.User?.DisplayName ?? string.Empty;
        var startLocal = appointment.StartDateTime.ToLocalTime();

        string subject = isConfirmation
            ? $"Підтвердження бронювання: {roomName}"
            : $"Скасування бронювання: {roomName}";

        string body = isConfirmation
            ? $"""
               <p>Шановний(-а) {appointment.ClientUser.DisplayName},</p>
               <p>Ваше бронювання <strong>{roomName}</strong> підтверджено.</p>
               <ul>
                 <li><strong>Дата:</strong> {startLocal:dd MMMM yyyy, HH:mm}</li>
                 <li><strong>Тривалість:</strong> {appointment.DurationMinutes} хв</li>
                 {(responsibleName != string.Empty ? $"<li><strong>Відповідальний:</strong> {responsibleName}</li>" : "")}
                 {(appointment.Notes != null ? $"<li><strong>Примітка:</strong> {appointment.Notes}</li>" : "")}
               </ul>
               <p>Деталі додано до вашого календаря Outlook.</p>
               """
            : $"""
               <p>Шановний(-а) {appointment.ClientUser.DisplayName},</p>
               <p>Ваше бронювання <strong>{roomName}</strong> скасовано.</p>
               {(appointment.CancellationReason != null ? $"<p><strong>Причина:</strong> {appointment.CancellationReason}</p>" : "")}
               """;

        await SendRawEmailAsync(recipientEmail, subject, body);
    }

    // Finds or creates "Бронювання приміщень" calendar for the given Azure AD user.
    // Propagates Graph API exceptions — callers must catch, log, and fall back.
    private async Task<string> GetOrCreateBookingsCalendarAsync(GraphServiceClient graph, string azureObjectId)
    {
        const string calendarName = "Бронювання приміщень";

        var response = await graph.Users[azureObjectId].Calendars.GetAsync();
        while (response != null)
        {
            if (response.Value != null)
            {
                var existing = response.Value.FirstOrDefault(c => c.Name == calendarName);
                if (existing?.Id != null) return existing.Id;
            }
            if (response.OdataNextLink == null) break;
            response = await graph.Users[azureObjectId].Calendars
                .WithUrl(response.OdataNextLink).GetAsync();
        }

        var created = await graph.Users[azureObjectId].Calendars.PostAsync(new Calendar
        {
            Name = calendarName,
            Color = CalendarColor.LightGreen,
        });
        return created?.Id ?? throw new InvalidOperationException("Graph API returned a null calendar id");
    }

    private static Event BuildCalendarEvent(Appointment appointment) => new()
    {
        Subject = $"Бронювання: {appointment.Room.Name}",
        Body = new ItemBody
        {
            ContentType = BodyType.Text,
            Content = appointment.Notes ?? appointment.Room.Description ?? appointment.Room.Name,
        },
        Start = new DateTimeTimeZone
        {
            DateTime = appointment.StartDateTime.UtcDateTime.ToString("o"),
            TimeZone = "UTC",
        },
        End = new DateTimeTimeZone
        {
            DateTime = appointment.EndDateTime.UtcDateTime.ToString("o"),
            TimeZone = "UTC",
        },
    };

    // Creates a calendar event for the client user.
    // Returns the new event id (stored in appointment.OutlookEventId) or null on failure.
    // Also independently creates the event for the responsible staff member.
    private async Task<string?> AddCalendarEventAsync(Appointment appointment, AppDbContext db)
    {
        var graph = BuildGraphClient();
        var ev = BuildCalendarEvent(appointment);
        var clientAzureId = appointment.ClientUser.AzureObjectId;

        // Resolve dedicated calendar; fall back to default if lookup/creation fails.
        string? calendarId = null;
        try
        {
            calendarId = await GetOrCreateBookingsCalendarAsync(graph, clientAzureId);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Failed to get/create bookings calendar for client {AzureId}", clientAzureId);
            await LogNotificationAsync(db, appointment.ClientUserId, appointment.Id,
                NotificationType.Calendar, false, ex.Message);
        }

        // Create the event (calendar-scoped when available; default calendar as fallback).
        string? eventId = null;
        try
        {
            Event? created = calendarId != null
                ? await graph.Users[clientAzureId].Calendars[calendarId].Events.PostAsync(ev)
                : await graph.Users[clientAzureId].Calendar.Events.PostAsync(ev);
            eventId = created?.Id;
            await LogNotificationAsync(db, appointment.ClientUserId, appointment.Id,
                NotificationType.Calendar, true);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Failed to create calendar event for appointment {Id}", appointment.Id);
            await LogNotificationAsync(db, appointment.ClientUserId, appointment.Id,
                NotificationType.Calendar, false, ex.Message);
        }

        // Staff event creation is fully independent — failure must not affect the client path.
        await AddStaffCalendarEventAsync(graph, appointment, db);

        return eventId;
    }

    // Creates a calendar event for the responsible staff member of the room (if assigned).
    // Any failure is logged and swallowed — it must not affect the client confirmation flow.
    private async Task AddStaffCalendarEventAsync(GraphServiceClient graph, Appointment appointment, AppDbContext db)
    {
        var room = await db.Rooms
            .Include(r => r.ResponsiblePerson)
                .ThenInclude(s => s!.User)
            .FirstOrDefaultAsync(r => r.Id == appointment.RoomId);

        var staffUser = room?.ResponsiblePerson?.User;
        if (staffUser == null) return;

        var staffAzureId = staffUser.AzureObjectId;
        var staffUserId = staffUser.Id;
        var ev = BuildCalendarEvent(appointment);

        string? calendarId = null;
        try
        {
            calendarId = await GetOrCreateBookingsCalendarAsync(graph, staffAzureId);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Failed to get/create bookings calendar for staff {AzureId}", staffAzureId);
            await LogNotificationAsync(db, staffUserId, appointment.Id,
                NotificationType.Calendar, false, ex.Message);
        }

        try
        {
            Event? staffCreated = calendarId != null
                ? await graph.Users[staffAzureId].Calendars[calendarId].Events.PostAsync(ev)
                : await graph.Users[staffAzureId].Calendar.Events.PostAsync(ev);

            if (staffCreated?.Id != null)
            {
                var tracked = await db.Appointments.FindAsync(appointment.Id);
                if (tracked != null)
                {
                    tracked.OutlookStaffEventId = staffCreated.Id;
                    await db.SaveChangesAsync();
                }
            }

            await LogNotificationAsync(db, staffUserId, appointment.Id,
                NotificationType.Calendar, true);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Failed to create staff calendar event for appointment {Id}", appointment.Id);
            await LogNotificationAsync(db, staffUserId, appointment.Id,
                NotificationType.Calendar, false, ex.Message);
        }
    }

    private async Task RemoveCalendarEventAsync(Appointment appointment)
    {
        var graph = BuildGraphClient();

        if (appointment.OutlookEventId != null)
        {
            try
            {
                await graph.Users[appointment.ClientUser.AzureObjectId]
                    .Events[appointment.OutlookEventId].DeleteAsync();
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to remove client calendar event for appointment {Id}", appointment.Id);
            }
        }
        // var staffUser = room?.ResponsiblePerson?.User;

        var staffUser = appointment.Room?.ResponsiblePerson?.User;
        if (staffUser != null && appointment.OutlookStaffEventId != null)
        {
            try
            {
                await graph.Users[staffUser.AzureObjectId]
                    .Events[appointment.OutlookStaffEventId].DeleteAsync();
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to remove staff calendar event for appointment {Id}", appointment.Id);
            }
        }
    }

    private static async Task LogNotificationAsync(
        AppDbContext db, Guid userId, Guid? appointmentId,
        NotificationType type, bool success, string? errorMessage = null)
    {
        db.Notifications.Add(new Notification
        {
            UserId = userId,
            AppointmentId = appointmentId,
            Type = type,
            Status = success ? NotificationStatus.Sent : NotificationStatus.Failed,
            SentAt = DateTime.UtcNow,
            ErrorMessage = errorMessage,
        });
        await db.SaveChangesAsync();
    }
}
