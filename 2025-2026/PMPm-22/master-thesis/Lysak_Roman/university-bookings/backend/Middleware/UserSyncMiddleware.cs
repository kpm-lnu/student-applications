using Microsoft.EntityFrameworkCore;
using UniversityBookings.Data;
using UniversityBookings.Models;
using FuzzySharp;
using System.Linq;
using System.Collections.Generic;

namespace UniversityBookings.Middleware;

/// <summary>
/// After JWT validation succeeds, ensures a User record exists for the caller.
/// Updates DisplayName, Email, and LastLoginAt on every request.
/// </summary>
public class UserSyncMiddleware(RequestDelegate next)
{
    // ---- Keyword Lists ----
    // Staff keywords from your job_titles.txt (cover major variants)
    static readonly string[] StaffKeywords = {
    "асистент", "асистентка", "асистет", "асистет", "астет", "доцент", "професор",
    "інженер", "інженер-програміст", "бухгалтер", "лаборант", "викладач",
    "секретар", "завідувач", "директор", "начальник", "менеджер", "модератор",
    "завідуючий", "методист", "інспектор", "координатор", "адміністратор", "кафедра", // Remove 'адміністратор' if you wish!
    // Add other unique non-student academic or support titles found in job_titles.txt
};
    // Student keywords from your job_titles.txt
    static readonly string[] StudentKeywords = {
    "student", "студент", "student.", "студент.", "студентка", "студентство",
    "студент 1 курсу", "студент 2 курсу", "студентка", "аспірант", "бакалавр",
    // Add other unique student-related variants you find
};
    public async Task InvokeAsync(HttpContext context, AppDbContext db)
    {
        if (context.User.Identity?.IsAuthenticated == true)
        {
            var oid = context.User.FindFirst("oid")?.Value
                   ?? context.User.FindFirst("http://schemas.microsoft.com/identity/claims/objectidentifier")?.Value;

            if (oid != null)
            {
                var email = context.User.FindFirst("preferred_username")?.Value
                         ?? context.User.FindFirst("email")?.Value
                         ?? context.User.FindFirst("upn")?.Value
                         ?? context.User.FindFirst("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/upn")?.Value
                         ?? string.Empty;
                var displayName = context.User.FindFirst("name")?.Value ?? email;

                var user = await db.Users.FirstOrDefaultAsync(u => u.AzureObjectId == oid);
                if (user is null)
                {


                    // Fuzzy match helper (returns true if input close to any keyword)
                    bool FuzzyContains(string input, IEnumerable<string> keywords, int threshold = 80)
                    {
                        if (string.IsNullOrWhiteSpace(input)) return false;
                        var normalized = input.ToLowerInvariant().Trim();
                        return keywords.Any(kw => Fuzz.Ratio(normalized, kw.ToLowerInvariant()) >= threshold);
                    }

                    // ---- Get jobTitle from claims ----
                    var jobTitle = context.User.FindFirst("jobTitle")?.Value ?? "";

                    // ---- Auto-assign role ----
                    UserRole assignedRole = UserRole.Student; // Default (as before)
                    if (FuzzyContains(jobTitle, StaffKeywords))
                    {
                        assignedRole = UserRole.Staff; // Assign Staff if matched
                    }
                    else if (FuzzyContains(jobTitle, StudentKeywords))
                    {
                        assignedRole = UserRole.Student; // Assign Student if matched
                    }
                    // Never assign Admin automatically!

                    // ---- Log unmatched jobTitles for admin review (optional) ----
                    if (!FuzzyContains(jobTitle, StaffKeywords) && !FuzzyContains(jobTitle, StudentKeywords))
                    {
                        // Example: log to a database table or simple file for periodic review
                        Console.WriteLine($"Unmatched jobTitle on new user: '{jobTitle}' ({email})");
                        // Optionally, create an 'UnmappedJobTitles' table for this purpose.
                    }

                    // ---- Create user ----
                    user = new User
                    {
                        AzureObjectId = oid,
                        Email = email,
                        DisplayName = displayName,
                        Role = assignedRole,
                    };
                    db.Users.Add(user);
                }
                else
                {
                    user.Email = email;
                    user.DisplayName = displayName;
                    user.LastLoginAt = DateTime.UtcNow;
                }

                await db.SaveChangesAsync();
            }
        }

        await next(context);
    }
}
