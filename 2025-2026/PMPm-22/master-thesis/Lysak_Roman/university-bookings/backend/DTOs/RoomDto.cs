using UniversityBookings.Models;

namespace UniversityBookings.DTOs;

public record RoomTypeDto(Guid Id, string Name, string Label);

public record AddressDto(Guid Id, string Street);

public record RoomResponsiblePersonDto(
    Guid Id,
    string DisplayName
);

public record RoomDto(
    Guid Id,
    string Name,
    string? RoomNumber,
    RoomTypeDto? RoomType,
    AddressDto? Address,
    string? Description,
    int? Capacity,
    bool IsActive,
    RoomResponsiblePersonDto? ResponsiblePerson,
    List<AvailabilityDto> Availability,
    SlotMode SlotMode
);

public record TimeSlotDto(
    string StartTime,   // ISO 8601 UTC
    string EndTime,
    bool Available,
    bool IsHeld = false
);

public record CreateRoomRequest(
    string Name,
    string? RoomNumber,
    Guid? RoomTypeId,
    Guid? AddressId,
    string? Description,
    int? Capacity,
    bool IsActive,
    Guid? ResponsiblePersonId,
    SlotMode SlotMode = SlotMode.Interval
);

public record UpdateRoomRequest(
    string Name,
    string? RoomNumber,
    Guid? RoomTypeId,
    Guid? AddressId,
    string? Description,
    int? Capacity,
    bool IsActive,
    Guid? ResponsiblePersonId,
    SlotMode SlotMode = SlotMode.Interval
);

public record CreateRoomTypeRequest(string Name, string Label);
public record UpdateRoomTypeRequest(string Name, string Label);

public record CreateAddressRequest(string Street);
public record UpdateAddressRequest(string Street);
