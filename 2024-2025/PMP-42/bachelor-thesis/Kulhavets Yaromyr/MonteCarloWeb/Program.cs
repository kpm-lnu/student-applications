var builder = WebApplication.CreateBuilder(args);

// Connect service
builder.Services.AddScoped<MonteCarloWeb.Services.IntegrationService>();
builder.Services.AddControllers();

// Swagger (test)
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Service for index.html
app.UseDefaultFiles();
app.UseStaticFiles();

// Swagger (dev)
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// Routing
app.UseHttpsRedirection();
app.UseAuthorization();
app.MapControllers();

app.Run();
