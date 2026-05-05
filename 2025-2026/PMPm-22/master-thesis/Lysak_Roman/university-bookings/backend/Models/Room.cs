namespace UniversityBookings.Models;

public class Room
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string Name { get; set; } = string.Empty;
    public string? RoomNumber { get; set; }
    public Guid? RoomTypeId { get; set; }
    public Guid? AddressId { get; set; }
    public string? Description { get; set; }
    public int? Capacity { get; set; }
    public bool IsActive { get; set; } = true;
    public Guid? ResponsiblePersonId { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public SlotMode SlotMode { get; set; } = SlotMode.Interval;

    // Navigation
    public RoomType? RoomType { get; set; }
    public Address? Address { get; set; }
    public StaffMember? ResponsiblePerson { get; set; }
    public ICollection<Appointment> Appointments { get; set; } = [];
    public ICollection<Availability> Availabilities { get; set; } = [];
}
