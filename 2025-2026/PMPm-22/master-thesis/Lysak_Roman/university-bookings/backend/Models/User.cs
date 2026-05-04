namespace UniversityBookings.Models;

public enum UserRole { Student, Staff, Admin }

public class User
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string AzureObjectId { get; set; } = string.Empty;  // "oid" claim
    public string Email { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    public UserRole Role { get; set; } = UserRole.Student;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime LastLoginAt { get; set; } = DateTime.UtcNow;

    // Navigation
    public StaffMember? StaffMember { get; set; }
    public ICollection<Appointment> Appointments { get; set; } = [];
    public ICollection<Notification> Notifications { get; set; } = [];
}
