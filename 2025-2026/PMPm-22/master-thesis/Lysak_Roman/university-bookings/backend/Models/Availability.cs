namespace UniversityBookings.Models;

public class Availability
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid RoomId { get; set; }
    public int DayOfWeek { get; set; }   // 0 = Sunday … 6 = Saturday
    public TimeSpan StartTime { get; set; }
    public TimeSpan EndTime { get; set; }
    public List<int> AvailableParaIndices { get; set; } = [];

    // Navigation
    public Room Room { get; set; } = null!;
}
