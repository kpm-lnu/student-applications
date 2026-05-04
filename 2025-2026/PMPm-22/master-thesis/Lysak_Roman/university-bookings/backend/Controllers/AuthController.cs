using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;

namespace UniversityBookings.Controllers;

[ApiController]
[Route("api/auth")]
public class AuthController(AppDbContext db) : ControllerBase
{
    /// <summary>
    /// Called by frontend after MSAL login. UserSyncMiddleware has already upserted the user.
    /// Returns the current user's profile with role.
    /// </summary>
    [HttpPost("login")]
    [Authorize]
    public async Task<ActionResult<UserDto>> Login()
    {
        var oid = User.FindFirst("oid")?.Value
               ?? User.FindFirst("http://schemas.microsoft.com/identity/claims/objectidentifier")?.Value;

        if (oid is null) return Unauthorized();

        var user = await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == oid);
        if (user is null) return Unauthorized();

        return Ok(user.ToDto());
    }
}
