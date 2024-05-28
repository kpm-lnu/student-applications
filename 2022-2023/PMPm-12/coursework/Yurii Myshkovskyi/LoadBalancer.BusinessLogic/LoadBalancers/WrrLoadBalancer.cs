using LoadBalancerApi.BusinessLogic;
using System.Collections.Concurrent;

public class WrrLoadBalancer : ILoadBalancer
{
    private readonly IServerProxyFactory _serverProxyFactory;
    private List<ServerProxy> _servers;
    private List<double> _cumulativeWeights;
    private ConcurrentDictionary<DelayComplexityFunction, int> _complexityCallCounterDictionary = new ConcurrentDictionary<DelayComplexityFunction, int>();
    private int _errorCounter = 0;
    object _lock = new object();

    public WrrLoadBalancer(IServerProxyFactory serverProxyFactory)
    {
        _serverProxyFactory = serverProxyFactory;
        _servers = _serverProxyFactory.SetupServers();
        CalculateCumulativeWeights();
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

        double randomNumber = new Random().NextDouble() * _cumulativeWeights.Last();
        int selectedIndex = _cumulativeWeights.BinarySearch(randomNumber);

        if (selectedIndex < 0)
        {
            selectedIndex = ~selectedIndex;
        }

        ServerProxy selectedServer = _servers[selectedIndex];
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

    private void CalculateCumulativeWeights()
    {
        _cumulativeWeights = new List<double>(_servers.Count);
        double cumulativeWeight = 0;

        foreach (var server in _servers)
        {
            cumulativeWeight += server.GetConstantWeight();
            _cumulativeWeights.Add(cumulativeWeight);
        }
    }

    public void PrintReport()
    {
        Console.WriteLine("WRR Loadbalancer Report.");
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

