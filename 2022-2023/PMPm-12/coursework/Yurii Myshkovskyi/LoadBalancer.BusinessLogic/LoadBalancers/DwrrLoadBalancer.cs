using LoadBalancerApi.BusinessLogic;
using System.Collections.Concurrent;

public class DwrrLoadBalancer : ILoadBalancer
{
    private readonly List<ServerProxy> _servers;
    private readonly IServerProxyFactory _serverProxyFactory;
    private int _currentIndex = -1;
    private int _errorCounter = 0;
    object _lock = new object();

    private ConcurrentDictionary<DelayComplexityFunction, int> _complexityCallCounterDictionary = new ConcurrentDictionary<DelayComplexityFunction, int>();

    public DwrrLoadBalancer(IServerProxyFactory serverProxyFactory)
    {
        _serverProxyFactory = serverProxyFactory;
        _servers = _serverProxyFactory.SetupServers();
        DelayComplexityFunction[] colors = (DelayComplexityFunction[])Enum.GetValues(typeof(DelayComplexityFunction));
        foreach (DelayComplexityFunction color in colors)
        {
            _complexityCallCounterDictionary[color] = 0;
        }
    }

    public async Task<Response> ForwardRequestAsync(Request request)
    {
        int serverCount = _servers.Count;

        if (serverCount == 0)
        {
            throw new InvalidOperationException("No servers available to forward the request.");
        }

        int startIndex = (_currentIndex + 1) % serverCount;
        double currentWeight = -1;
        int selectedIndex = -1;

        for (int i = 0; i < serverCount; i++)
        {
            int index = (startIndex + i) % serverCount;
            ServerProxy server = _servers[index];
            double weight = server.GetActualWeight();

            if (weight > currentWeight)
            {
                currentWeight = weight;
                selectedIndex = index;
            }
        }

        if (selectedIndex == -1)
        {
            throw new InvalidOperationException("Failed to find a suitable server to forward the request.");
        }

        _currentIndex = selectedIndex;
        ServerProxy selectedServer = _servers[selectedIndex];
        lock (_lock)
        {
            _complexityCallCounterDictionary[selectedServer.DelayComplexity]++;
        }
        try
        {
            return await selectedServer.AcceptRequestAsync(request);
        }
        catch (Exception ex)
        {
            Interlocked.Increment(ref _errorCounter);
            return new Response { IsSuccess = false };
        }
    }

    public void PrintReport()
    {
        Console.WriteLine("DWRR Loadbalancer Report.");
        Console.WriteLine($"Servers Count: {_servers.Count}");
        Console.WriteLine($"Errors Count: {_errorCounter}");
        int sum = 0;
        foreach (var record in _complexityCallCounterDictionary)
        {
            sum += record.Value;
            Console.WriteLine($"Server Delay Function: {record.Key}, Server Requests Count: {record.Value}");
        }
        Console.WriteLine($"Total: {sum}");
    }
}
