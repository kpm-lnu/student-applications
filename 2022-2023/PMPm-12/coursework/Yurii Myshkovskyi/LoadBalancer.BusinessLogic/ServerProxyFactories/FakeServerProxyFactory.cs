using LoadBalancerApi.BusinessLogic;

namespace LoadBalancerApi.ServerProxyFactories
{
    public class FakeServerProxyFactory : IServerProxyFactory
    {
        const int SERVERS_COUNT = 16;
        public List<ServerProxy> SetupServers()
        {
            var servers = new List<ServerProxy>();
            for (int i = 0; i < SERVERS_COUNT; ++i)
            {
                var delayComplexityFunction = (DelayComplexityFunction)(i % 8);
                servers.Add(new ServerProxy(delayComplexityFunction));
            }
            return servers;
        }
    }
}
