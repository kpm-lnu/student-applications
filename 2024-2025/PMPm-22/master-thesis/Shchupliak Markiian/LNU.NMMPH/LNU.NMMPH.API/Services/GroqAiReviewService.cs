using LNU.NMMPH.API.Interface;

using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace LNU.NMMPH.API.Services
{
    public class GroqAiReviewService : IGroqAiReviewService
    {
        private readonly string _apiKey;

        public GroqAiReviewService(IConfiguration config)
        {
            _apiKey = "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";
        }

        public async Task<string> ReviewCodeAsync(string code, string method)
        {
            try
            {
                HttpClient httpClient = new();

                var prompt = $"""
                    You are an expert assistant evaluating student solutions for numerical methods.

                    Here is the student's implementation of the method {method}:

                    {code}

                    Please:
                    - Check if the student correctly implements the {method} method.
                    - Identify any algorithmic or logical mistakes.
                    - Suggest improvements to structure.
                    - Do NOT describe what the {method} method is and refactored code. Only analyze the code.
                    """;


                var requestData = new
                {
                    model = "meta-llama/llama-4-scout-17b-16e-instruct",
                    messages = new[]
                    {
                        new { role = "system", content = "You are a university assistant who checks if student code implements numerical methods correctly." },
                        new { role = "user", content = prompt }
                    },
                    temperature = 0.3,
                    //max_tokens = 1000
                };

                var request = new HttpRequestMessage(HttpMethod.Post, "https://api.groq.com/openai/v1/chat/completions")
                {
                    Content = new StringContent(JsonSerializer.Serialize(requestData), Encoding.UTF8, "application/json")
                };

                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

                var response = await httpClient.SendAsync(request);
                string json = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    return null;
                }

                using var doc = JsonDocument.Parse(json);
                string result = doc.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString();

                return result!;
            }
            catch (Exception ex)
            {
                return null;
            }
        }
    }
}
