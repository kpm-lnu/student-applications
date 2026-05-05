using Microsoft.EntityFrameworkCore;
using UniversityBookings.Models;

namespace UniversityBookings.Data;

public class AppDbContext(DbContextOptions<AppDbContext> options) : DbContext(options)
{
    public DbSet<User> Users => Set<User>();
    public DbSet<Room> Rooms => Set<Room>();
    public DbSet<RoomType> RoomTypes => Set<RoomType>();
    public DbSet<Address> Addresses => Set<Address>();
    public DbSet<StaffMember> StaffMembers => Set<StaffMember>();
    public DbSet<Appointment> Appointments => Set<Appointment>();
    public DbSet<Availability> Availabilities => Set<Availability>();
    public DbSet<Notification> Notifications => Set<Notification>();
    public DbSet<SlotHold> SlotHolds => Set<SlotHold>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        base.OnModelCreating(b);

        // User
        b.Entity<User>(e =>
        {
            e.HasIndex(u => u.AzureObjectId).IsUnique();
            e.HasIndex(u => u.Email).IsUnique();
            e.Property(u => u.Role).HasConversion<string>();
        });

        // RoomType
        b.Entity<RoomType>(e =>
        {
            e.HasIndex(rt => rt.Name).IsUnique();
        });

        // Address — no extra config needed

        // Room
        b.Entity<Room>(e =>
        {
            e.HasOne(r => r.RoomType)
             .WithMany(rt => rt.Rooms)
             .HasForeignKey(r => r.RoomTypeId)
             .OnDelete(DeleteBehavior.SetNull);

            e.HasOne(r => r.Address)
             .WithMany(a => a.Rooms)
             .HasForeignKey(r => r.AddressId)
             .OnDelete(DeleteBehavior.SetNull);

            e.HasOne(r => r.ResponsiblePerson)
             .WithMany(sm => sm.Rooms)
             .HasForeignKey(r => r.ResponsiblePersonId)
             .OnDelete(DeleteBehavior.SetNull);
        });

        // StaffMember
        b.Entity<StaffMember>(e =>
        {
            e.HasOne(sm => sm.User)
             .WithOne(u => u.StaffMember)
             .HasForeignKey<StaffMember>(sm => sm.UserId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        // Appointment
        b.Entity<Appointment>(e =>
        {
            e.Property(a => a.Status).HasConversion<string>();
            e.HasOne(a => a.Room)
             .WithMany(r => r.Appointments)
             .HasForeignKey(a => a.RoomId)
             .OnDelete(DeleteBehavior.Restrict);
            e.HasOne(a => a.ClientUser)
             .WithMany(u => u.Appointments)
             .HasForeignKey(a => a.ClientUserId)
             .OnDelete(DeleteBehavior.Restrict);
        });

        // Availability
        b.Entity<Availability>(e =>
        {
            e.Property(av => av.AvailableParaIndices)
             .HasConversion(
                 v => System.Text.Json.JsonSerializer.Serialize(v, (System.Text.Json.JsonSerializerOptions?)null),
                 v => System.Text.Json.JsonSerializer.Deserialize<List<int>>(v, (System.Text.Json.JsonSerializerOptions?)null) ?? new List<int>())
             .HasColumnType("nvarchar(max)");

            e.HasOne(av => av.Room)
             .WithMany(r => r.Availabilities)
             .HasForeignKey(av => av.RoomId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        // SlotHold
        b.Entity<SlotHold>(e =>
        {
            e.HasOne(h => h.Room)
             .WithMany()
             .HasForeignKey(h => h.RoomId)
             .OnDelete(DeleteBehavior.Cascade);
            e.HasOne(h => h.User)
             .WithMany()
             .HasForeignKey(h => h.UserId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        // Notification
        b.Entity<Notification>(e =>
        {
            e.Property(n => n.Type).HasConversion<string>();
            e.Property(n => n.Status).HasConversion<string>();
            e.HasOne(n => n.User)
             .WithMany(u => u.Notifications)
             .HasForeignKey(n => n.UserId)
             .OnDelete(DeleteBehavior.Cascade);
            e.HasOne(n => n.Appointment)
             .WithMany(a => a.Notifications)
             .HasForeignKey(n => n.AppointmentId)
             .OnDelete(DeleteBehavior.Cascade);
        });
    }
}
