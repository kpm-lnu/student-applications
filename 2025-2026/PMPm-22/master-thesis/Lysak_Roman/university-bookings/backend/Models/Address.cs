namespace UniversityBookings.Models;

public class Address
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string Street { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public ICollection<Room> Rooms { get; set; } = [];
}
