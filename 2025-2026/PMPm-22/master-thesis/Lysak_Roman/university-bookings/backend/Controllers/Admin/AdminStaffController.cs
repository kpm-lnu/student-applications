using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;

namespace UniversityBookings.Controllers.Admin;

[ApiController]
[Route("api/admin/staff")]
[Authorize(Policy = "AdminOnly")]
public class AdminStaffController(AppDbContext db) : ControllerBase
{
    [HttpGet]
    public async Task<ActionResult<IEnumerable<StaffMemberDto>>> GetAll()
    {
        var staff = await db.StaffMembers
            .Include(sm => sm.User)
            .OrderBy(sm => sm.User.DisplayName)
            .ToListAsync();
        return Ok(staff.Select(sm => sm.ToDto()));
    }

    [HttpGet("{id:guid}")]
    public async Task<ActionResult<StaffMemberDto>> GetById(Guid id)
    {
        var sm = await db.StaffMembers
            .Include(sm => sm.User)
            .FirstOrDefaultAsync(sm => sm.Id == id);
        if (sm is null) return NotFound();
        return Ok(sm.ToDto());
    }
}
