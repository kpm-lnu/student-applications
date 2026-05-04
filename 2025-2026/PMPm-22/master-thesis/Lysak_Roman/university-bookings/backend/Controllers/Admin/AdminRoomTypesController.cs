using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Models;

namespace UniversityBookings.Controllers.Admin;

[ApiController]
[Route("api/admin/room-types")]
[Authorize(Policy = "AdminOnly")]
public class AdminRoomTypesController(AppDbContext db) : ControllerBase
{
    [HttpGet]
    public async Task<ActionResult<IEnumerable<RoomTypeDto>>> GetAll()
    {
        var types = await db.RoomTypes.OrderBy(rt => rt.Label).ToListAsync();
        return Ok(types.Select(rt => rt.ToDto()));
    }

    [HttpGet("{id:guid}")]
    public async Task<ActionResult<RoomTypeDto>> GetById(Guid id)
    {
        var rt = await db.RoomTypes.FindAsync(id);
        if (rt is null) return NotFound();
        return Ok(rt.ToDto());
    }

    [HttpPost]
    public async Task<ActionResult<RoomTypeDto>> Create([FromBody] CreateRoomTypeRequest req)
    {
        if (await db.RoomTypes.AnyAsync(rt => rt.Name == req.Name.ToLowerInvariant()))
            return Conflict("Тип з таким slug вже існує.");

        var rt = new RoomType
        {
            Name = req.Name.ToLowerInvariant().Trim(),
            Label = req.Label.Trim(),
        };
        db.RoomTypes.Add(rt);
        await db.SaveChangesAsync();
        return CreatedAtAction(nameof(GetById), new { id = rt.Id }, rt.ToDto());
    }

    [HttpPut("{id:guid}")]
    public async Task<ActionResult<RoomTypeDto>> Update(Guid id, [FromBody] UpdateRoomTypeRequest req)
    {
        var rt = await db.RoomTypes.FindAsync(id);
        if (rt is null) return NotFound();

        var nameNorm = req.Name.ToLowerInvariant().Trim();
        if (await db.RoomTypes.AnyAsync(x => x.Name == nameNorm && x.Id != id))
            return Conflict("Тип з таким slug вже існує.");

        rt.Name = nameNorm;
        rt.Label = req.Label.Trim();
        await db.SaveChangesAsync();
        return Ok(rt.ToDto());
    }

    [HttpDelete("{id:guid}")]
    public async Task<IActionResult> Delete(Guid id)
    {
        var rt = await db.RoomTypes.FindAsync(id);
        if (rt is null) return NotFound();

        bool inUse = await db.Rooms.AnyAsync(r => r.RoomTypeId == id);
        if (inUse)
            return Conflict("Неможливо видалити тип: він використовується у приміщеннях.");

        db.RoomTypes.Remove(rt);
        await db.SaveChangesAsync();
        return NoContent();
    }
}
