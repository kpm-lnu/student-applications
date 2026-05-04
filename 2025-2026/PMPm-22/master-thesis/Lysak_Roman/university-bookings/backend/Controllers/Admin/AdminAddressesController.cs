using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Models;

namespace UniversityBookings.Controllers.Admin;

[ApiController]
[Route("api/admin/addresses")]
[Authorize(Policy = "AdminOnly")]
public class AdminAddressesController(AppDbContext db) : ControllerBase
{
    [HttpGet]
    public async Task<ActionResult<IEnumerable<AddressDto>>> GetAll()
    {
        var addresses = await db.Addresses.OrderBy(a => a.Street).ToListAsync();
        return Ok(addresses.Select(a => a.ToDto()));
    }

    [HttpGet("{id:guid}")]
    public async Task<ActionResult<AddressDto>> GetById(Guid id)
    {
        var addr = await db.Addresses.FindAsync(id);
        if (addr is null) return NotFound();
        return Ok(addr.ToDto());
    }

    [HttpPost]
    public async Task<ActionResult<AddressDto>> Create([FromBody] CreateAddressRequest req)
    {
        var addr = new Address { Street = req.Street.Trim() };
        db.Addresses.Add(addr);
        await db.SaveChangesAsync();
        return CreatedAtAction(nameof(GetById), new { id = addr.Id }, addr.ToDto());
    }

    [HttpPut("{id:guid}")]
    public async Task<ActionResult<AddressDto>> Update(Guid id, [FromBody] UpdateAddressRequest req)
    {
        var addr = await db.Addresses.FindAsync(id);
        if (addr is null) return NotFound();
        addr.Street = req.Street.Trim();
        await db.SaveChangesAsync();
        return Ok(addr.ToDto());
    }

    [HttpDelete("{id:guid}")]
    public async Task<IActionResult> Delete(Guid id)
    {
        var addr = await db.Addresses.FindAsync(id);
        if (addr is null) return NotFound();

        bool inUse = await db.Rooms.AnyAsync(r => r.AddressId == id);
        if (inUse)
            return Conflict("Неможливо видалити адресу: вона використовується у приміщеннях.");

        db.Addresses.Remove(addr);
        await db.SaveChangesAsync();
        return NoContent();
    }
}
