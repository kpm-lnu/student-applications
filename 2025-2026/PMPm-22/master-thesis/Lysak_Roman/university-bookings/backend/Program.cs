using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Authorization;
using Microsoft.EntityFrameworkCore;
using Microsoft.Identity.Web;
using UniversityBookings.Auth;
using UniversityBookings.Data;
using UniversityBookings.Hubs;
using UniversityBookings.Middleware;
using UniversityBookings.Services;

var builder = WebApplication.CreateBuilder(args);

// ── Authentication (Azure AD / Microsoft.Identity.Web) ────────────────────────
builder.Services.AddMicrosoftIdentityWebApiAuthentication(builder.Configuration, "AzureAd");

// SignalR sends JWT via ?access_token= query param (WebSocket can't set headers)
builder.Services.PostConfigure<JwtBearerOptions>(JwtBearerDefaults.AuthenticationScheme, options =>
{
    var original = options.Events.OnMessageReceived;
    options.Events.OnMessageReceived = async context =>
    {
        if (original != null) await original(context);
        var token = context.Request.Query["access_token"];
        if (!string.IsNullOrEmpty(token) &&
            context.HttpContext.Request.Path.StartsWithSegments("/hubs"))
            context.Token = token;
    };
});

// ── Authorization policies ────────────────────────────────────────────────────
// AdminOnly is backed by a DB role check via AdminAuthorizationHandler
builder.Services.AddScoped<IAuthorizationHandler, AdminAuthorizationHandler>();
builder.Services.AddAuthorizationBuilder()
    .AddPolicy("AdminOnly", policy =>
        policy.RequireAuthenticatedUser()
              .AddRequirements(new AdminRequirement()));

// ── Database ───────────────────────────────────────────────────────────────────
builder.Services.AddDbContext<AppDbContext>(opt =>
    opt.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

// ── SignalR ────────────────────────────────────────────────────────────────────
builder.Services.AddSignalR();

// ── CORS ───────────────────────────────────────────────────────────────────────
var allowedOrigins = builder.Configuration.GetSection("AllowedOrigins").Get<string[]>()
    ?? ["http://localhost:3000"];

builder.Services.AddCors(opt =>
    opt.AddDefaultPolicy(p =>
        p.WithOrigins(allowedOrigins)
         .AllowAnyHeader()
         .WithMethods("GET", "POST", "PATCH", "DELETE")
         .AllowCredentials())); // needed for SignalR

// ── App Services ───────────────────────────────────────────────────────────────
builder.Services.AddHttpClient();
builder.Services.AddScoped<GeminiService>();
builder.Services.AddScoped<AppointmentService>();
builder.Services.AddScoped<AvailabilityService>();
builder.Services.AddScoped<NotificationService>();
builder.Services.AddScoped<AppointmentHubService>();
builder.Services.AddScoped<SlotHoldService>();
builder.Services.AddHostedService<HoldCleanupService>();

// ── Controllers + Swagger ──────────────────────────────────────────────────────
builder.Services.AddControllers()
    .AddJsonOptions(opt =>
        opt.JsonSerializerOptions.Converters.Add(
            new System.Text.Json.Serialization.JsonStringEnumConverter()));
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(opt =>
{
    opt.SwaggerDoc("v1", new() { Title = "University Bookings API", Version = "v1" });
    opt.AddSecurityDefinition("Bearer", new Microsoft.OpenApi.Models.OpenApiSecurityScheme
    {
        Name = "Authorization",
        Type = Microsoft.OpenApi.Models.SecuritySchemeType.Http,
        Scheme = "bearer",
        BearerFormat = "JWT",
        In = Microsoft.OpenApi.Models.ParameterLocation.Header,
        Description = "Enter your Azure AD Bearer token",
    });
    opt.AddSecurityRequirement(new Microsoft.OpenApi.Models.OpenApiSecurityRequirement
    {
        {
            new Microsoft.OpenApi.Models.OpenApiSecurityScheme
            {
                Reference = new Microsoft.OpenApi.Models.OpenApiReference
                {
                    Type = Microsoft.OpenApi.Models.ReferenceType.SecurityScheme,
                    Id = "Bearer",
                },
            },
            Array.Empty<string>()
        }
    });
});

// ── Build ──────────────────────────────────────────────────────────────────────
var app = builder.Build();

// ── DB Migrations + Seed (development only) ───────────────────────────────────
if (app.Environment.IsDevelopment())
{
    using var scope = app.Services.CreateScope();
    var dbCtx = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    dbCtx.Database.Migrate();
    await DbSeeder.SeedAsync(dbCtx);

    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors();
app.UseAuthentication();
app.UseMiddleware<UserSyncMiddleware>();
app.UseAuthorization();
app.MapControllers();
app.MapHub<AppointmentHub>("/hubs/appointments");

app.Run();
