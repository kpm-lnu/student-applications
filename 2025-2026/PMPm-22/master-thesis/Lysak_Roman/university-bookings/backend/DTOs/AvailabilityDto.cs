namespace UniversityBookings.DTOs;

public record AvailabilityDto(
    Guid Id,
    Guid RoomId,
    int DayOfWeek,              // 0 = Sunday … 6 = Saturday
    string StartTime,           // HH:mm
    string EndTime,             // HH:mm
    List<int> AvailableParaIndices
);

public record SetAvailabilityRequest(
    int DayOfWeek,
    string StartTime,           // HH:mm
    string EndTime,             // HH:mm
    List<int>? AvailableParaIndices = null
);
