using UniversityBookings.Models;

namespace UniversityBookings.DTOs;

public record AppointmentDto(
    Guid Id,
    Guid RoomId,
    string RoomName,
    string RoomType,
    Guid ClientUserId,
    UserDto? ClientUser,
    string StartDateTime,
    string EndDateTime,
    int DurationMinutes,
    AppointmentStatus Status,
    string? Notes,
    string CreatedAt,
    string? CancelledAt,
    string? CancellationReason
);

public record CreateAppointmentRequest(
    Guid RoomId,
    int DurationMinutes,    // must be one of: 40, 60, 80, 120
    string StartDateTime,   // ISO 8601
    string? Notes
);

public record CancelAppointmentRequest(string? Reason);

public record UpdateAppointmentStatusRequest(AppointmentStatus Status);

public record DashboardStatsDto(
    int TotalBookingsToday,
    int TotalBookingsThisMonth,
    int PendingAppointments,
    double CancellationRate,
    IEnumerable<PopularRoomDto> PopularRooms,
    IEnumerable<BookingsPerDayDto> BookingsPerDay
);

public record PopularRoomDto(Guid RoomId, string RoomName, int Count);
public record BookingsPerDayDto(string Date, int Count);
