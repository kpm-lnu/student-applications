using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace UniversityBookings.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddRoomTypesAndAddresses : Migration
    {
        // Known GUIDs for seed data so the data migration SQL can reference them
        private const string ClassroomTypeId  = "A1000000-0000-0000-0000-000000000001";
        private const string SportTypeId      = "A2000000-0000-0000-0000-000000000002";
        private const string ConferenceTypeId = "A3000000-0000-0000-0000-000000000003";
        private const string Address1Id       = "B1000000-0000-0000-0000-000000000001";
        private const string Address2Id       = "B2000000-0000-0000-0000-000000000002";

        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            // ── Create RoomTypes ──────────────────────────────────────────────
            migrationBuilder.CreateTable(
                name: "RoomTypes",
                columns: table => new
                {
                    Id        = table.Column<Guid>(type: "uniqueidentifier", nullable: false),
                    Name      = table.Column<string>(type: "nvarchar(450)", nullable: false),
                    Label     = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: false),
                },
                constraints: table => { table.PrimaryKey("PK_RoomTypes", x => x.Id); });

            migrationBuilder.CreateIndex(
                name: "IX_RoomTypes_Name",
                table: "RoomTypes",
                column: "Name",
                unique: true);

            // ── Seed 3 room types ─────────────────────────────────────────────
            migrationBuilder.Sql($@"
                INSERT INTO RoomTypes (Id, Name, Label, CreatedAt) VALUES
                ('{ClassroomTypeId}',  'classroom',  N'Аудиторія',       GETUTCDATE()),
                ('{SportTypeId}',      'sport',      N'Спортивний зал',  GETUTCDATE()),
                ('{ConferenceTypeId}', 'conference', N'Конференц-зал',   GETUTCDATE());
            ");

            // ── Create Addresses ──────────────────────────────────────────────
            migrationBuilder.CreateTable(
                name: "Addresses",
                columns: table => new
                {
                    Id        = table.Column<Guid>(type: "uniqueidentifier", nullable: false),
                    Street    = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: false),
                },
                constraints: table => { table.PrimaryKey("PK_Addresses", x => x.Id); });

            // ── Seed 2 addresses ──────────────────────────────────────────────
            migrationBuilder.Sql($@"
                INSERT INTO Addresses (Id, Street, CreatedAt) VALUES
                ('{Address1Id}', N'вул. Університетська, 1', GETUTCDATE()),
                ('{Address2Id}', N'вул. Черемшини, 31',      GETUTCDATE());
            ");

            // ── Add FK columns to Rooms ───────────────────────────────────────
            migrationBuilder.AddColumn<Guid>(
                name: "RoomTypeId",
                table: "Rooms",
                type: "uniqueidentifier",
                nullable: true);

            migrationBuilder.AddColumn<Guid>(
                name: "AddressId",
                table: "Rooms",
                type: "uniqueidentifier",
                nullable: true);

            // ── Data migration: assign room types based on old Type string ────
            migrationBuilder.Sql($@"
                UPDATE Rooms SET RoomTypeId = '{ClassroomTypeId}'
                WHERE Type = 'Classroom';

                UPDATE Rooms SET RoomTypeId = '{SportTypeId}'
                WHERE Type IN ('SportsBig', 'SportsSmall');

                UPDATE Rooms SET AddressId = '{Address1Id}'
                WHERE Type = 'Classroom';

                UPDATE Rooms SET AddressId = '{Address2Id}'
                WHERE Type IN ('SportsBig', 'SportsSmall');
            ");

            // ── Drop old Type column ──────────────────────────────────────────
            migrationBuilder.DropColumn(name: "Type", table: "Rooms");

            // ── Add FK constraints ────────────────────────────────────────────
            migrationBuilder.CreateIndex(
                name: "IX_Rooms_RoomTypeId",
                table: "Rooms",
                column: "RoomTypeId");

            migrationBuilder.CreateIndex(
                name: "IX_Rooms_AddressId",
                table: "Rooms",
                column: "AddressId");

            migrationBuilder.AddForeignKey(
                name: "FK_Rooms_RoomTypes_RoomTypeId",
                table: "Rooms",
                column: "RoomTypeId",
                principalTable: "RoomTypes",
                principalColumn: "Id",
                onDelete: ReferentialAction.SetNull);

            migrationBuilder.AddForeignKey(
                name: "FK_Rooms_Addresses_AddressId",
                table: "Rooms",
                column: "AddressId",
                principalTable: "Addresses",
                principalColumn: "Id",
                onDelete: ReferentialAction.SetNull);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(name: "FK_Rooms_RoomTypes_RoomTypeId", table: "Rooms");
            migrationBuilder.DropForeignKey(name: "FK_Rooms_Addresses_AddressId", table: "Rooms");
            migrationBuilder.DropIndex(name: "IX_Rooms_RoomTypeId", table: "Rooms");
            migrationBuilder.DropIndex(name: "IX_Rooms_AddressId", table: "Rooms");

            // Restore Type column
            migrationBuilder.AddColumn<string>(
                name: "Type",
                table: "Rooms",
                type: "nvarchar(max)",
                nullable: false,
                defaultValue: "Classroom");

            migrationBuilder.Sql($@"
                UPDATE Rooms SET Type = 'Classroom'
                WHERE RoomTypeId = '{ClassroomTypeId}';

                UPDATE Rooms SET Type = 'SportsBig'
                WHERE RoomTypeId = '{SportTypeId}';
            ");

            migrationBuilder.DropColumn(name: "RoomTypeId", table: "Rooms");
            migrationBuilder.DropColumn(name: "AddressId", table: "Rooms");

            migrationBuilder.DropTable(name: "RoomTypes");
            migrationBuilder.DropTable(name: "Addresses");
        }
    }
}
