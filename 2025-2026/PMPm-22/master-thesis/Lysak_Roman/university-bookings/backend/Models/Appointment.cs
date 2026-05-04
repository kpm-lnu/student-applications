namespace UniversityBookings.Models;

public enum AppointmentStatus { Pending, Confirmed, Cancelled, Completed }

public class Appointment
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid RoomId { get; set; }
    public Guid ClientUserId { get; set; }
    public DateTimeOffset StartDateTime { get; set; }
    public DateTimeOffset EndDateTime { get; set; }
    public int DurationMinutes { get; set; }
    public AppointmentStatus Status { get; set; } = AppointmentStatus.Pending;
    public string? Notes { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? CancelledAt { get; set; }
    public string? CancellationReason { get; set; }
    public string? OutlookEventId { get; set; }
    public string? OutlookStaffEventId { get; set; }

    // Navigation
    public Room Room { get; set; } = null!;
    public User ClientUser { get; set; } = null!;
    public ICollection<Notification> Notifications { get; set; } = [];
}
