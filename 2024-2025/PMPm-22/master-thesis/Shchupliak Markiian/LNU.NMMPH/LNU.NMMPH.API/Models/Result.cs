namespace LNU.NMMPH.API.Models
{
    public class Result<T>
    {
        public string AiReview { get; set; } = string.Empty; 
        public T Value { get; set; } = default!;  
    }
}
