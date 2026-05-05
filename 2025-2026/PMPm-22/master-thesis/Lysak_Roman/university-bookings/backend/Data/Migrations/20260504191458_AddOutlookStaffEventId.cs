using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace UniversityBookings.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddOutlookStaffEventId : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "OutlookStaffEventId",
                table: "Appointments",
                type: "nvarchar(max)",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "OutlookStaffEventId",
                table: "Appointments");
        }
    }
}
