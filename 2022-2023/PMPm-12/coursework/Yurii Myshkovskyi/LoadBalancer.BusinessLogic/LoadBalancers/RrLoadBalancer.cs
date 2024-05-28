using LoadBalancerApi.BusinessLogic;
using System.Collections.Concurrent;

public class RrLoadBalancer : ILoadBalancer
{
    private readonly IServerProxyFactory _serverProxyFactory;
    private List<ServerProxy> _servers;
    private int _currentIndex;
    private int _errorCounter;
    private int _requestCounter = 0;
    object _lock = new object();

    private Dictionary<DelayComplexityFunction, int> _complexityCallCounterDictionary = new Dictionary<DelayComplexityFunction, int>();

    public RrLoadBalancer(IServerProxyFactory serverProxyFactory)
    {
        _serverProxyFactory = serverProxyFactory;
        _servers = _serverProxyFactory.SetupServers();
        _currentIndex = -1;
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

        _currentIndex = (_currentIndex + 1) % serverCount;
        ServerProxy selectedServer = _servers[_currentIndex];
        lock (_lock)
        {
            _complexityCallCounterDictionary[selectedServer.DelayComplexity]++;
        }
        try
        {
            return await selectedServer.AcceptRequestAsync(request);
        }
        catch
        {
            Interlocked.Increment(ref _errorCounter);
            return new Response { IsSuccess = false };
        }
    }

    public void PrintReport()
    {
        Console.WriteLine("RR Loadbalancer Report.");
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


