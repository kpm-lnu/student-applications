using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.DTOs;
using UniversityBookings.Models;

namespace UniversityBookings.Services;

public record ChatMessageDto(string Role, string Text);

public class GeminiService(
    IConfiguration config,
    IHttpClientFactory httpClientFactory,
    AppDbContext db,
    AppointmentService appointmentService,
    AvailabilityService availabilityService,
    SlotHoldService holdService)
{
    private const string GeminiModel = "gemini-2.5-flash";
    private const int MaxToolIterations = 5;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public async Task<(string Text, bool AppointmentChanged)> ChatAsync(User user, ChatMessageDto[] history, string userMessage)
    {
        var apiKey = config["Gemini:ApiKey"]
            ?? throw new InvalidOperationException("Gemini:ApiKey not configured.");

        var client = httpClientFactory.CreateClient();

        var contents = history
            .Select(m => new GeminiContent(m.Role, [new GeminiPart { Text = m.Text }]))
            .Concat([new GeminiContent("user", [new GeminiPart { Text = userMessage }])])
            .ToList();

        bool appointmentChanged = false;

        for (int i = 0; i < MaxToolIterations; i++)
        {
            var request = new GeminiRequest(
                Contents: [.. contents],
                SystemInstruction: new GeminiSystemInstruction([new GeminiPart { Text = BuildSystemPrompt(user) }]),
                Tools: [new GeminiTool(ToolDeclarations)]
            );

            var response = await CallGeminiAsync(client, apiKey, request);
            var candidate = response.Candidates?.FirstOrDefault()?.Content;
            if (candidate is null) return ("", appointmentChanged);

            var functionCalls = candidate.Parts
                .Where(p => p.FunctionCall is not null)
                .Select(p => p.FunctionCall!)
                .ToList();

            if (functionCalls.Count == 0)
                return (candidate.Parts.FirstOrDefault(p => p.Text is not null)?.Text ?? "", appointmentChanged);

            // Append model's function-call turn to history
            contents.Add(new GeminiContent("model", [.. candidate.Parts]));

            // Execute all tool calls, collect results
            var resultParts = new List<GeminiPart>();
            foreach (var call in functionCalls)
            {
                object result;
                try
                {
                    result = await ExecuteToolAsync(user, call.Name, call.Args);
                    if (call.Name is "createAppointment" or "cancelAppointment")
                        appointmentChanged = true;
                }
                catch (Exception ex)
                {
                    result = new { error = ex.Message };
                }
                resultParts.Add(new GeminiPart
                {
                    FunctionResponse = new GeminiFunctionResponse(call.Name, new { result })
                });
            }
            contents.Add(new GeminiContent("user", [.. resultParts]));
        }

        return ("", appointmentChanged);
    }

    // ── HTTP call to Gemini REST API ──────────────────────────────────────────

    private static async Task<GeminiResponse> CallGeminiAsync(
        HttpClient client, string apiKey, GeminiRequest request)
    {
        var url = $"https://generativelanguage.googleapis.com/v1beta/models/{GeminiModel}:generateContent?key={apiKey}";
        var body = new StringContent(
            JsonSerializer.Serialize(request, JsonOptions),
            System.Text.Encoding.UTF8,
            "application/json");

        var httpResponse = await client.PostAsync(url, body);
        httpResponse.EnsureSuccessStatusCode();

        var json = await httpResponse.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<GeminiResponse>(json, JsonOptions)
            ?? throw new InvalidOperationException("Empty response from Gemini.");
    }

    // ── Tool executor ─────────────────────────────────────────────────────────

    private async Task<object> ExecuteToolAsync(User user, string toolName, JsonElement args)
    {
        switch (toolName)
        {
            case "searchRooms":
            {
                string? type = args.TryGetProperty("type", out var t) ? t.GetString() : null;
                var rooms = await db.Rooms
                    .Include(r => r.RoomType)
                    .Include(r => r.Address)
                    .Include(r => r.ResponsiblePerson).ThenInclude(sm => sm!.User)
                    .Include(r => r.Availabilities)
                    .Where(r => r.IsActive && (type == null || r.RoomType!.Name == type))
                    .OrderBy(r => r.Name)
                    .ToListAsync();
                return rooms.Select(r => r.ToDto());
            }

            case "getUserAppointments":
            {
                var list = await db.Appointments
                    .Include(a => a.Room).ThenInclude(r => r!.RoomType)
                    .Include(a => a.ClientUser)
                    .Where(a => a.ClientUserId == user.Id &&
                                a.Status != AppointmentStatus.Cancelled &&
                                a.Status != AppointmentStatus.Completed)
                    .OrderBy(a => a.StartDateTime)
                    .ToListAsync();
                return list.Select(a => a.ToDto());
            }

            case "getRoomAvailability":
            {
                var roomId = Guid.Parse(args.GetProperty("roomId").GetString()!);
                var date = args.GetProperty("date").GetString()!;
                var duration = args.TryGetProperty("duration", out var dEl) && dEl.GetString() is { } dStr
                    ? int.Parse(dStr) : 80;
                return await availabilityService.GetAvailableSlotsAsync(roomId, date, duration, user.Id);
            }

            case "createAppointment":
            {
                var roomId = Guid.Parse(args.GetProperty("roomId").GetString()!);
                var rawDt = args.GetProperty("startDateTime").GetString()!;
                var durationMinutes = args.TryGetProperty("durationMinutes", out var dmEl) && dmEl.GetString() is { } dmStr
                    ? int.Parse(dmStr) : 80;
                string? notes = args.TryGetProperty("notes", out var n) ? n.GetString() : null;

                var startUtc = KyivLocalToUtc(rawDt).ToString("o");
                var req = new CreateAppointmentRequest(roomId, durationMinutes, startUtc, notes);
                var appointment = await appointmentService.CreateAsync(user.Id, req);
                await holdService.ReleaseByUserAndRoomAsync(user.Id, roomId);
                return appointment.ToDto();
            }

            case "cancelAppointment":
            {
                var appointmentId = Guid.Parse(args.GetProperty("appointmentId").GetString()!);
                await appointmentService.CancelAsync(appointmentId, user.Id, false, null);
                return new { success = true, message = "Бронювання успішно скасовано." };
            }

            case "getAdminAppointments" when user.Role == UserRole.Admin:
            {
                var query = db.Appointments
                    .Include(a => a.Room).ThenInclude(r => r!.RoomType)
                    .Include(a => a.ClientUser)
                    .AsQueryable();

                if (args.TryGetProperty("status", out var sEl) && sEl.GetString() is { } sStr
                    && Enum.TryParse<AppointmentStatus>(sStr, out var st))
                    query = query.Where(a => a.Status == st);

                if (args.TryGetProperty("from", out var fEl) && fEl.GetString() is { } fStr
                    && DateOnly.TryParse(fStr, out var from))
                    query = query.Where(a => a.StartDateTime >= new DateTimeOffset(from.ToDateTime(TimeOnly.MinValue), TimeSpan.Zero));

                if (args.TryGetProperty("to", out var tEl) && tEl.GetString() is { } tStr
                    && DateOnly.TryParse(tStr, out var to))
                    query = query.Where(a => a.StartDateTime < new DateTimeOffset(to.AddDays(1).ToDateTime(TimeOnly.MinValue), TimeSpan.Zero));

                var list = await query.OrderByDescending(a => a.StartDateTime).ToListAsync();
                return list.Select(a => a.ToDto());
            }

            case "updateAppointmentStatus" when user.Role == UserRole.Admin:
            {
                var appointmentId = Guid.Parse(args.GetProperty("appointmentId").GetString()!);
                var status = Enum.Parse<AppointmentStatus>(args.GetProperty("status").GetString()!);
                var appointment = await appointmentService.UpdateStatusAsync(appointmentId, status);
                return appointment.ToDto();
            }

            case "getDashboardStats" when user.Role == UserRole.Admin:
            {
                var now = DateTime.UtcNow;
                var monthStart = new DateTime(now.Year, now.Month, 1, 0, 0, 0, DateTimeKind.Utc);

                var allThisMonth = await db.Appointments
                    .Where(a => a.CreatedAt >= monthStart)
                    .ToListAsync();

                int totalToday = allThisMonth.Count(a => a.CreatedAt.Date == now.Date);
                int totalMonth = allThisMonth.Count;
                int pending = await db.Appointments.CountAsync(a => a.Status == AppointmentStatus.Pending);
                int cancelled = allThisMonth.Count(a => a.Status == AppointmentStatus.Cancelled);
                double cancelRate = totalMonth > 0 ? (double)cancelled / totalMonth * 100 : 0;

                var roomIds = allThisMonth
                    .Where(a => a.Status != AppointmentStatus.Cancelled)
                    .Select(a => a.RoomId).Distinct().ToList();
                var roomNames = await db.Rooms
                    .Where(r => roomIds.Contains(r.Id))
                    .Select(r => new { r.Id, r.Name })
                    .ToDictionaryAsync(r => r.Id, r => r.Name);

                var popularRooms = allThisMonth
                    .Where(a => a.Status != AppointmentStatus.Cancelled)
                    .GroupBy(a => a.RoomId)
                    .Select(g => new
                    {
                        roomId = g.Key,
                        name = roomNames.GetValueOrDefault(g.Key, "Unknown"),
                        count = g.Count()
                    })
                    .OrderByDescending(p => p.count).Take(5);

                return new { totalToday, totalMonth, pending, cancelRate, popularRooms };
            }

            default:
                return new { error = $"Невідома або заборонена функція: {toolName}" };
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static DateTimeOffset KyivLocalToUtc(string localDateTimeStr)
    {
        // Strip any tz suffix the model may have added
        var bare = localDateTimeStr.TrimEnd('Z').Split('+')[0].Split('-') is { Length: > 3 } parts
            ? string.Join("-", parts[..3])
            : localDateTimeStr.TrimEnd('Z').Split('+')[0];

        if (!DateTime.TryParse(localDateTimeStr.TrimEnd('Z').Split('+')[0], out var local))
            return DateTimeOffset.UtcNow;

        var kyiv = TimeZoneInfo.FindSystemTimeZoneById("FLE Standard Time");
        return TimeZoneInfo.ConvertTimeToUtc(local, kyiv);
    }

    private static string BuildSystemPrompt(User user)
    {
        var kyiv = TimeZoneInfo.FindSystemTimeZoneById("FLE Standard Time");
        var now = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, kyiv)
            .ToString("dddd, dd MMMM yyyy HH:mm", new System.Globalization.CultureInfo("uk-UA"));

        return $"""
            Ти — помічник університетської системи бронювання приміщень. Відповідай лише українською мовою. Будь ввічливим та корисним.

            Поточна дата та час: {now}

            Ім'я користувача: {user.DisplayName}
            Email: {user.Email}
            Роль: {user.Role}

            --- Режими слотів ---
            Приміщення мають поле slotMode: "Interval" або "Para".

            Interval (вільні слоти): користувач обирає тривалість 40/60/80/120 хв. При виклику getRoomAvailability передавай duration (наприклад "60"). При createAppointment передавай durationMinutes.

            Para (пари): фіксовані блоки занять — тривалість завжди 80 хв. Розклад пар:
            Пара 1: 08:30–09:50
            Пара 2: 10:10–11:30
            Пара 3: 11:50–13:10
            Пара 4: 13:30–14:50
            Пара 5: 15:05–16:25
            Пара 6: 16:40–18:00
            Пара 7: 18:10–19:30
            Пара 8: 19:40–21:00
            Для Para-приміщень: при getRoomAvailability duration не потрібен (пропусти або передай "80"). При createAppointment durationMinutes не потрібен (пропусти або передай "80"). startDateTime має збігатися з точним часом початку пари (наприклад "2026-05-10T08:30:00").

            --- Правила поведінки ---
            1. Для отримання інформації про приміщення або бронювання завжди використовуй доступні інструменти — не вигадуй дані самостійно.
            2. Перед скасуванням бронювання обов'язково запитай підтвердження у користувача. Ніколи не скасовуй автоматично.
            3. Якщо інструмент повернув помилку — повідом користувача коректно, не намагайся вигадати відповідь.
            4. Відповідай виключно на питання, пов'язані з бронюванням приміщень університету.
            5. Ніколи не показуй технічні ідентифікатори (UUID, id) користувачу — вони лише для внутрішнього використання інструментів.
            6. Не використовуй markdown-розмітку (зірочки, решітки, підкреслення). Відповідай простим текстом. Для переліків використовуй тире або нумерацію без зайвих символів.
            7. Мінімізуй кількість інструментів: не запитуй дані, які вже є у контексті.
            8. Якщо користувач вказав конкретні кімнату, дату та час — дій напряму: searchRooms → createAppointment (без проміжного getRoomAvailability). Якщо API поверне помилку про зайнятість — тоді виклич getRoomAvailability і запропонуй вільні слоти.
            9. Після успішного createAppointment одразу повертай текстову відповідь — НЕ викликай getUserAppointments для перевірки.
            """;
    }

    // ── Tool declarations ─────────────────────────────────────────────────────

    private static readonly object[] ToolDeclarations =
    [
        new
        {
            name = "searchRooms",
            description = "Повертає список доступних приміщень університету. Якщо type не вказано — всі приміщення. Викликай ОДИН РАЗ і використовуй результат.",
            parameters = new
            {
                type = "object",
                properties = new
                {
                    type = new
                    {
                        type = "string",
                        description = "Slug типу: \"classroom\" — аудиторія, \"sport\" — спортивний зал, \"conference\" — конференц-зал.",
                        @enum = new[] { "classroom", "sport", "conference" }
                    }
                }
            }
        },
        new
        {
            name = "getUserAppointments",
            description = "Повертає активні бронювання поточного користувача."
        },
        new
        {
            name = "getRoomAvailability",
            description = "Повертає вільні часові слоти для конкретного приміщення на вказану дату. Для Para-приміщень duration не потрібен.",
            parameters = new
            {
                type = "object",
                properties = new
                {
                    roomId = new { type = "string", description = "Ідентифікатор приміщення (UUID)." },
                    date = new { type = "string", description = "Дата у форматі YYYY-MM-DD." },
                    duration = new
                    {
                        type = "string",
                        description = "Тривалість у хвилинах (тільки для Interval-приміщень).",
                        @enum = new[] { "40", "60", "80", "120" }
                    }
                },
                required = new[] { "roomId", "date" }
            }
        },
        new
        {
            name = "createAppointment",
            description = "Створює нове бронювання приміщення. Для Para-приміщень durationMinutes не потрібен (80 хв автоматично). Після успішного створення одразу повідом користувача — НЕ викликай getUserAppointments для підтвердження.",
            parameters = new
            {
                type = "object",
                properties = new
                {
                    roomId = new { type = "string", description = "Ідентифікатор приміщення (UUID)." },
                    startDateTime = new
                    {
                        type = "string",
                        description = "Дата та час початку за київським часом у форматі \"YYYY-MM-DDTHH:MM:SS\" — БЕЗ суфікса Z. Для Para-приміщень має збігатися з точним часом початку пари."
                    },
                    durationMinutes = new
                    {
                        type = "string",
                        description = "Тривалість у хвилинах (тільки для Interval-приміщень).",
                        @enum = new[] { "40", "60", "80", "120" }
                    },
                    notes = new { type = "string", description = "Примітки (необов'язково)." }
                },
                required = new[] { "roomId", "startDateTime" }
            }
        },
        new
        {
            name = "cancelAppointment",
            description = "Скасовує бронювання користувача. Викликати лише після явного підтвердження від користувача.",
            parameters = new
            {
                type = "object",
                properties = new
                {
                    appointmentId = new { type = "string", description = "Ідентифікатор бронювання (UUID)." }
                },
                required = new[] { "appointmentId" }
            }
        },
        new
        {
            name = "getAdminAppointments",
            description = "Тільки для адміністратора. Повертає всі бронювання з можливістю фільтрації.",
            parameters = new
            {
                type = "object",
                properties = new
                {
                    status = new
                    {
                        type = "string",
                        @enum = new[] { "Pending", "Confirmed", "Cancelled", "Completed" }
                    },
                    from = new { type = "string", description = "YYYY-MM-DD" },
                    to = new { type = "string", description = "YYYY-MM-DD" }
                }
            }
        },
        new
        {
            name = "updateAppointmentStatus",
            description = "Тільки для адміністратора. Змінює статус бронювання.",
            parameters = new
            {
                type = "object",
                properties = new
                {
                    appointmentId = new { type = "string", description = "UUID бронювання." },
                    status = new
                    {
                        type = "string",
                        @enum = new[] { "Pending", "Confirmed", "Cancelled", "Completed" }
                    }
                },
                required = new[] { "appointmentId", "status" }
            }
        },
        new
        {
            name = "getDashboardStats",
            description = "Тільки для адміністратора. Повертає статистику системи."
        }
    ];
}

// ── Gemini REST API models ────────────────────────────────────────────────────

public record GeminiRequest(
    [property: JsonPropertyName("contents")] GeminiContent[] Contents,
    [property: JsonPropertyName("systemInstruction")] GeminiSystemInstruction? SystemInstruction = null,
    [property: JsonPropertyName("tools")] GeminiTool[]? Tools = null
);

public record GeminiSystemInstruction(
    [property: JsonPropertyName("parts")] GeminiPart[] Parts
);

public record GeminiTool(
    [property: JsonPropertyName("functionDeclarations")] object[] FunctionDeclarations
);

public record GeminiContent(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("parts")] GeminiPart[] Parts
);

public class GeminiPart
{
    [JsonPropertyName("text")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Text { get; init; }

    [JsonPropertyName("functionCall")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public GeminiFunctionCall? FunctionCall { get; init; }

    [JsonPropertyName("functionResponse")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public GeminiFunctionResponse? FunctionResponse { get; init; }
}

public record GeminiFunctionCall(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("args")] JsonElement Args
);

public record GeminiFunctionResponse(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("response")] object Response
);

public record GeminiResponse(
    [property: JsonPropertyName("candidates")] GeminiCandidate[]? Candidates
);

public record GeminiCandidate(
    [property: JsonPropertyName("content")] GeminiContent Content
);
