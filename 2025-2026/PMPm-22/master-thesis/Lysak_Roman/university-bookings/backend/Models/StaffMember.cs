namespace UniversityBookings.Models;

public class StaffMember
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid UserId { get; set; }

    // Navigation
    public User User { get; set; } = null!;
    public ICollection<Room> Rooms { get; set; } = [];
}
