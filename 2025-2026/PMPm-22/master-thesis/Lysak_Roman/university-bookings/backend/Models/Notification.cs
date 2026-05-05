namespace UniversityBookings.Models;

public enum NotificationType { Email, Teams, Calendar }
public enum NotificationStatus { Sent, Failed }

public class Notification
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid UserId { get; set; }
    public Guid? AppointmentId { get; set; }
    public NotificationType Type { get; set; }
    public DateTime SentAt { get; set; } = DateTime.UtcNow;
    public NotificationStatus Status { get; set; }
    public string? ErrorMessage { get; set; }

    // Navigation
    public User User { get; set; } = null!;
    public Appointment? Appointment { get; set; }
}
