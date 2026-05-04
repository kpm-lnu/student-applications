using UniversityBookings.Models;

namespace UniversityBookings.DTOs;

public record UserDto(
    Guid Id,
    string AzureObjectId,
    string Email,
    string DisplayName,
    UserRole Role,
    DateTime CreatedAt,
    DateTime LastLoginAt
);

public record UpdateRoleRequest(UserRole Role);

public record StaffMemberDto(
    Guid Id,
    Guid UserId,
    string DisplayName,
    string Email
);
