using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace UniversityBookings.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddParaSlotMode : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "SlotMode",
                table: "Rooms",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<string>(
                name: "AvailableParaIndices",
                table: "Rooms",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "[]");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "SlotMode",
                table: "Rooms");

            migrationBuilder.DropColumn(
                name: "AvailableParaIndices",
                table: "Rooms");
        }
    }
}
