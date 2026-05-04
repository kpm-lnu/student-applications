using Microsoft.EntityFrameworkCore;
using UniversityBookings.Models;

namespace UniversityBookings.Data;

public static class DbSeeder
{
    public static async Task SeedAsync(AppDbContext db)
    {
        if (await db.Rooms.AnyAsync()) return;

        // ── Users (reuse existing seed users if already present) ──────────────
        var adminUser = await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == "seed-admin-oid");
        if (adminUser is null)
        {
            adminUser = new User
            {
                AzureObjectId = "seed-admin-oid",
                Email = "admin@university.edu.ua",
                DisplayName = "Адміністратор Системи",
                Role = UserRole.Admin,
            };
            db.Users.Add(adminUser);
        }

        var staffUser1 = await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == "seed-staff1-oid");
        if (staffUser1 is null)
        {
            staffUser1 = new User
            {
                AzureObjectId = "seed-staff1-oid",
                Email = "librarian@university.edu.ua",
                DisplayName = "Олена Коваленко",
                Role = UserRole.Staff,
            };
            db.Users.Add(staffUser1);
        }

        var staffUser2 = await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == "seed-staff2-oid");
        if (staffUser2 is null)
        {
            staffUser2 = new User
            {
                AzureObjectId = "seed-staff2-oid",
                Email = "it.support@university.edu.ua",
                DisplayName = "Іван Мельник",
                Role = UserRole.Staff,
            };
            db.Users.Add(staffUser2);
        }

        await db.SaveChangesAsync();

        // ── StaffMembers (reuse existing if already present) ──────────────────
        var sm1 = await db.StaffMembers.FirstOrDefaultAsync(sm => sm.UserId == staffUser1.Id);
        if (sm1 is null)
        {
            sm1 = new StaffMember { UserId = staffUser1.Id };
            db.StaffMembers.Add(sm1);
        }

        var sm2 = await db.StaffMembers.FirstOrDefaultAsync(sm => sm.UserId == staffUser2.Id);
        if (sm2 is null)
        {
            sm2 = new StaffMember { UserId = staffUser2.Id };
            db.StaffMembers.Add(sm2);
        }

        await db.SaveChangesAsync();

        // ── RoomTypes (inserted by migration, just look them up) ──────────────
        var classroomType = await db.RoomTypes.FirstOrDefaultAsync(rt => rt.Name == "classroom");
        var sportType     = await db.RoomTypes.FirstOrDefaultAsync(rt => rt.Name == "sport");

        // ── Addresses (inserted by migration, just look them up) ──────────────
        var addr1 = await db.Addresses.FirstOrDefaultAsync(a => a.Street == "вул. Університетська, 1");
        var addr2 = await db.Addresses.FirstOrDefaultAsync(a => a.Street == "вул. Черемшини, 31");

        // ── Rooms ─────────────────────────────────────────────────────────────
        var rooms = new List<Room>
        {
            new Room
            {
                Name = "Аудиторія 101",
                RoomNumber = "101",
                RoomTypeId = classroomType?.Id,
                AddressId = addr1?.Id,
                Description = "Навчальна аудиторія на першому поверсі головного корпусу.",
                Capacity = 30,
                ResponsiblePersonId = sm1.Id,
            },
            new Room
            {
                Name = "Аудиторія 102",
                RoomNumber = "102",
                RoomTypeId = classroomType?.Id,
                AddressId = addr1?.Id,
                Description = "Навчальна аудиторія на першому поверсі головного корпусу.",
                Capacity = 30,
                ResponsiblePersonId = sm1.Id,
            },
            new Room
            {
                Name = "Аудиторія 201",
                RoomNumber = "201",
                RoomTypeId = classroomType?.Id,
                AddressId = addr1?.Id,
                Description = "Навчальна аудиторія на другому поверсі головного корпусу.",
                Capacity = 40,
                ResponsiblePersonId = sm1.Id,
            },
            new Room
            {
                Name = "Аудиторія 202",
                RoomNumber = "202",
                RoomTypeId = classroomType?.Id,
                AddressId = addr1?.Id,
                Description = "Навчальна аудиторія на другому поверсі головного корпусу.",
                Capacity = 40,
                ResponsiblePersonId = sm1.Id,
            },
            new Room
            {
                Name = "Аудиторія 301",
                RoomNumber = "301",
                RoomTypeId = classroomType?.Id,
                AddressId = addr1?.Id,
                Description = "Навчальна аудиторія на третьому поверсі головного корпусу.",
                Capacity = 50,
                ResponsiblePersonId = sm1.Id,
            },
            new Room
            {
                Name = "Спортивний зал (великий)",
                RoomNumber = null,
                RoomTypeId = sportType?.Id,
                AddressId = addr2?.Id,
                Description = "Великий спортивний зал для командних ігор та масових заходів.",
                Capacity = 200,
                ResponsiblePersonId = sm2.Id,
            },
            new Room
            {
                Name = "Спортивний зал (малий)",
                RoomNumber = null,
                RoomTypeId = sportType?.Id,
                AddressId = addr2?.Id,
                Description = "Малий спортивний зал для індивідуальних тренувань та секцій.",
                Capacity = 50,
                ResponsiblePersonId = sm2.Id,
            },
        };

        db.Rooms.AddRange(rooms);
        await db.SaveChangesAsync();

        // ── Availability (Mon–Fri 08:00–20:00 for each room) ──────────────────
        foreach (var room in rooms)
        {
            for (int d = 1; d <= 5; d++)
            {
                db.Availabilities.Add(new Availability
                {
                    RoomId = room.Id,
                    DayOfWeek = d,
                    StartTime = new TimeSpan(8, 0, 0),
                    EndTime = new TimeSpan(20, 0, 0),
                });
            }
        }

        await db.SaveChangesAsync();
    }
}
