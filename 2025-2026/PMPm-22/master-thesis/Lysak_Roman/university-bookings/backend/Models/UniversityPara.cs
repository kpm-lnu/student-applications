namespace UniversityBookings.Models;

public static class UniversityPara
{
    public static readonly (int Index, TimeSpan Start, TimeSpan End)[] Schedule =
    [
        (1, new TimeSpan(8, 30, 0),  new TimeSpan(9, 50, 0)),
        (2, new TimeSpan(10, 10, 0), new TimeSpan(11, 30, 0)),
        (3, new TimeSpan(11, 50, 0), new TimeSpan(13, 10, 0)),
        (4, new TimeSpan(13, 30, 0), new TimeSpan(14, 50, 0)),
        (5, new TimeSpan(15, 5, 0),  new TimeSpan(16, 25, 0)),
        (6, new TimeSpan(16, 40, 0), new TimeSpan(18, 0, 0)),
        (7, new TimeSpan(18, 10, 0), new TimeSpan(19, 30, 0)),
        (8, new TimeSpan(19, 40, 0), new TimeSpan(21, 0, 0)),
    ];

    public static bool IsValidPara(TimeSpan localStart, TimeSpan localEnd) =>
        Schedule.Any(p => p.Start == localStart && p.End == localEnd);
}
