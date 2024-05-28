
// Add services to the container.


// Configure the HTTP request pipeline.





public interface ILoadBalancer
{
    Task<Response> ForwardRequestAsync(Request request);
    void PrintReport();
}

