using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.Models;

namespace UniversityBookings.Hubs;

[Authorize]
public class AppointmentHub(AppDbContext db) : Hub
{
    // Groups: "admins" for admin-wide broadcasts, "user-{dbUserId}" per user

    public override async Task OnConnectedAsync()
    {
        var oid = Context.User?.FindFirst("oid")?.Value
               ?? Context.User?.FindFirst("http://schemas.microsoft.com/identity/claims/objectidentifier")?.Value;

        if (oid != null)
        {
            var user = await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == oid);
            if (user != null)
            {
                // Group key matches what AppointmentHubService.NotifyUser() uses
                await Groups.AddToGroupAsync(Context.ConnectionId, $"user-{user.Id}");

                if (user.Role == UserRole.Admin)
                    await Groups.AddToGroupAsync(Context.ConnectionId, "admins");
            }
        }

        await base.OnConnectedAsync();
    }
}

/// <summary>
/// Helper service injected into controllers to push SignalR events.
/// </summary>
public class AppointmentHubService(IHubContext<AppointmentHub> hub)
{
    public Task NotifyAdminsNewAppointment(object payload) =>
        hub.Clients.Group("admins").SendAsync("NewAppointmentCreated", payload);

    public Task NotifyAdminsStatusChanged(object payload) =>
        hub.Clients.Group("admins").SendAsync("AppointmentStatusChanged", payload);

    public Task NotifyUser(string userId, string eventName, object payload) =>
        hub.Clients.Group($"user-{userId}").SendAsync(eventName, payload);
}
