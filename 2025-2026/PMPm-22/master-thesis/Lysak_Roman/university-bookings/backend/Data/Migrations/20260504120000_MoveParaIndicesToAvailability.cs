using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace UniversityBookings.Data.Migrations
{
    /// <inheritdoc />
    public partial class MoveParaIndicesToAvailability : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "AvailableParaIndices",
                table: "Rooms");

            migrationBuilder.AddColumn<string>(
                name: "AvailableParaIndices",
                table: "Availabilities",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "[]");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "AvailableParaIndices",
                table: "Availabilities");

            migrationBuilder.AddColumn<string>(
                name: "AvailableParaIndices",
                table: "Rooms",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "[]");
        }
    }
}
