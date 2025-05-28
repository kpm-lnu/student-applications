
namespace LNU.NMMPH.API.Interface
{
    public interface IGroqAiReviewService
    {
        Task<string> ReviewCodeAsync(string code, string method);
    }
}
