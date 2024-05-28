namespace LoadBalancerApi.BusinessLogic
{
    public interface IServerProxyFactory
    {
        List<ServerProxy> SetupServers();
    }
}
