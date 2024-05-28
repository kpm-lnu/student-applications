using LoadBalancerApi.BusinessLogic;
using LoadBalancerApi.ServerProxyFactories;

namespace LoadBalancerApi
{
    public static class DependencyInjectionSetupExtension
    {
        public static void SetupDependencyInjection(this IServiceCollection services)
        {
            services.AddSingleton<ILoadBalancer, DwrrLoadBalancer>();
            services.AddSingleton<IServerProxyFactory, FakeServerProxyFactory>();
        }
    }
}
