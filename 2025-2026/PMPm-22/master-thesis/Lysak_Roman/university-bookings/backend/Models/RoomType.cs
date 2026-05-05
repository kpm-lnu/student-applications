namespace UniversityBookings.Models;

public class RoomType
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string Name { get; set; } = string.Empty;    // slug(ярлик): "classroom", "sport", "conference"
    public string Label { get; set; } = string.Empty;   // display: "Аудиторія", etc.
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public ICollection<Room> Rooms { get; set; } = [];
}
