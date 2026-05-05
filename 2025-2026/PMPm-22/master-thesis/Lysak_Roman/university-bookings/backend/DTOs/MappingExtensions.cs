using UniversityBookings.Models;

namespace UniversityBookings.DTOs;

public static class MappingExtensions
{
    public static UserDto ToDto(this User u) => new(
        u.Id, u.AzureObjectId, u.Email, u.DisplayName, u.Role,
        u.CreatedAt, u.LastLoginAt);

    public static StaffMemberDto ToDto(this StaffMember sm) => new(
        sm.Id, sm.UserId,
        sm.User?.DisplayName ?? string.Empty,
        sm.User?.Email ?? string.Empty);

    public static AvailabilityDto ToDto(this Availability av) => new(
        av.Id, av.RoomId, av.DayOfWeek,
        av.StartTime.ToString(@"hh\:mm"),
        av.EndTime.ToString(@"hh\:mm"),
        av.AvailableParaIndices);

    public static RoomTypeDto ToDto(this RoomType rt) => new(rt.Id, rt.Name, rt.Label);

    public static AddressDto ToDto(this Address a) => new(a.Id, a.Street);

    public static RoomDto ToDto(this Room r) => new(
        r.Id, r.Name, r.RoomNumber,
        r.RoomType?.ToDto(),
        r.Address?.ToDto(),
        r.Description, r.Capacity, r.IsActive,
        r.ResponsiblePerson is null ? null : new RoomResponsiblePersonDto(
            r.ResponsiblePerson.Id,
            r.ResponsiblePerson.User?.DisplayName ?? string.Empty),
        r.Availabilities.Select(av => av.ToDto()).OrderBy(av => av.DayOfWeek).ToList(),
        r.SlotMode);

    public static AppointmentDto ToDto(this Appointment a) => new(
        a.Id, a.RoomId,
        a.Room?.Name ?? string.Empty,
        a.Room?.RoomType?.Label ?? string.Empty,
        a.ClientUserId,
        a.ClientUser?.ToDto(),
        a.StartDateTime.ToString("o"),
        a.EndDateTime.ToString("o"),
        a.DurationMinutes,
        a.Status,
        a.Notes,
        a.CreatedAt.ToString("o"),
        a.CancelledAt?.ToString("o"),
        a.CancellationReason);
}
