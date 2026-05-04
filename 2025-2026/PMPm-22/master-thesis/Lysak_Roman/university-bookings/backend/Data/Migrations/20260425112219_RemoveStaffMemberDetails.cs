using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace UniversityBookings.Data.Migrations
{
    /// <inheritdoc />
    public partial class RemoveStaffMemberDetails : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Bio",
                table: "StaffMembers");

            migrationBuilder.DropColumn(
                name: "Department",
                table: "StaffMembers");

            migrationBuilder.DropColumn(
                name: "Position",
                table: "StaffMembers");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "Bio",
                table: "StaffMembers",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "Department",
                table: "StaffMembers",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "Position",
                table: "StaffMembers",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "");
        }
    }
}
