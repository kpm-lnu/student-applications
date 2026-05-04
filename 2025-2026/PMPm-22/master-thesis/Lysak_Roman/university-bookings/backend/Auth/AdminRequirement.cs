using Microsoft.AspNetCore.Authorization;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.Models;

namespace UniversityBookings.Auth;

public class AdminRequirement : IAuthorizationRequirement { }

public class AdminAuthorizationHandler(AppDbContext db) : AuthorizationHandler<AdminRequirement>
{
    protected override async Task HandleRequirementAsync(
        AuthorizationHandlerContext context, AdminRequirement requirement)
    {
        var oid = context.User.FindFirst("oid")?.Value
               ?? context.User.FindFirst("http://schemas.microsoft.com/identity/claims/objectidentifier")?.Value;

        if (oid is null) return;

        var user = await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == oid);
        if (user?.Role == UserRole.Admin)
            context.Succeed(requirement);
    }
}
