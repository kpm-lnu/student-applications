using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;

namespace UniversityBookings.Controllers;

[ApiController]
[Route("api/room-types")]
public class RoomTypesController(AppDbContext db) : ControllerBase
{
    [HttpGet]
    public async Task<ActionResult<IEnumerable<RoomTypeDto>>> GetAll()
    {
        var types = await db.RoomTypes.OrderBy(rt => rt.Label).ToListAsync();
        return Ok(types.Select(rt => rt.ToDto()));
    }
}
