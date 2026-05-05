# EF Core Migrations

## Initial setup (run from /backend directory):

```bash
# Install EF Core tools globally (if not already)
dotnet tool install --global dotnet-ef

# Create initial migration
dotnet ef migrations add InitialCreate --output-dir Data/Migrations

# Apply to database
dotnet ef database update
```

## Creating subsequent migrations:

```bash
dotnet ef migrations add <MigrationName> --output-dir Data/Migrations
dotnet ef database update
```

## Notes:
- Never use `Database.EnsureCreated()` in production — always use Migrate()
- Seed data is in `DbSeeder.cs` and runs only in Development
- Production deployments should run `dotnet ef database update` in CI/CD pipeline
