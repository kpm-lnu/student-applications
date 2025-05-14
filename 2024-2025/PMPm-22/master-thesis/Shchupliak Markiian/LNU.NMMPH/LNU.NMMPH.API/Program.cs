using LNU.NMMPH.API.Interface;
using LNU.NMMPH.API.Interface.Methods;
using LNU.NMMPH.API.Services;
using LNU.NMMPH.API.Services.Methods;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
builder.Services.AddScoped<IMethodsService, MethodsService>();
builder.Services.AddScoped<IEulerMethod, EulerMethod>();
builder.Services.AddScoped<IRungeKuttaMethod, RungeKuttaMethod>();
builder.Services.AddScoped<IPoissonMethod, PoissonMethod>();
builder.Services.AddScoped<IGroqAiReviewService, GroqAiReviewService>();

builder.Services.AddCors(options =>
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins("http://localhost:4200");
        policy.AllowCredentials();
        policy.AllowAnyMethod();
        policy.AllowAnyHeader();
    }));

// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();
app.UseCors();

app.MapControllers();

app.Run();
