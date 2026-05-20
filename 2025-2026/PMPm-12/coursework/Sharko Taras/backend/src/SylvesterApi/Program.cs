using SylvesterApi.Contracts;
using SylvesterApi.Services;
using SylvesterApi.Validation;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddOpenApi();
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
        policy.AllowAnyOrigin()
            .AllowAnyHeader()
            .AllowAnyMethod());
});

builder.Services.AddScoped<IMatrixValidator, MatrixValidator>();
builder.Services.AddScoped<ISylvesterSolverService, SylvesterSolverService>();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseCors();

app.MapGet("/api/health", () => Results.Ok(new { Status = "ok" }));

app.MapPost("/api/sylvester/solve",
    (SolveSylvesterRequest request, IMatrixValidator validator, ISylvesterSolverService solverService) =>
    {
        var errors = validator.Validate(request);
        if (errors.Count > 0)
        {
            return Results.BadRequest(new { Errors = errors });
        }

        try
        {
            var response = solverService.Solve(request);
            return Results.Ok(response);
        }
        catch (ArgumentException ex)
        {
            return Results.BadRequest(new { Errors = new[] { ex.Message } });
        }
        catch (Exception ex)
        {
            return Results.Problem(
                detail: ex.Message,
                title: "Failed to solve Sylvester equation",
                statusCode: StatusCodes.Status500InternalServerError);
        }
    })
    .WithName("SolveSylvesterEquation");

app.Run();
