using LoadBalancerApi.ServerProxyFactories;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace LoadBalancerSpammer
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var serversFactory = new FakeServerProxyFactory();

            var configs = new List<Tuple<int, int>>()
            {
                new Tuple<int, int>(50, 100),
                new Tuple<int, int>(25, 400),
                new Tuple<int, int>(50, 500)
            };

            foreach (var config in configs)
            {
                var loadBalancers = new ILoadBalancer[]
                {
                    new DwrrLoadBalancer(serversFactory),
                    //new WrrLoadBalancer(serversFactory),
                    //new RrLoadBalancer(serversFactory)
                };
                int parallelRequests = config.Item1;
                int requestsPerSender = config.Item2;
                foreach (var loadBalancer in loadBalancers)
                {
                    Stopwatch stopwatch = Stopwatch.StartNew();
                    List<Task> tasks = new List<Task>(parallelRequests);
                    for (int i = 0; i < parallelRequests; i++)
                    {
                        tasks.Add(SendRequestsAsync(loadBalancer, requestsPerSender));
                    }

                    await Task.WhenAll(tasks);
                    stopwatch.Stop();
                    Console.WriteLine("_______________________________________________________________________________________");
                    Console.WriteLine($"Parallel Senders: {parallelRequests}, Requests per each Sender: {requestsPerSender}, Total Requests: {parallelRequests * requestsPerSender}");
                    Console.WriteLine($"Execution Time:{stopwatch.ElapsedMilliseconds}");
                    loadBalancer.PrintReport();
                }
            }
            Console.WriteLine();
            Console.WriteLine("Execution Finished");
            Console.ReadKey();
        }

        private static async Task SendRequestsAsync(ILoadBalancer loadBalancer, int requestPerSender)
        {
            int requestCounter = 0;
            while (requestCounter < requestPerSender)
            {
                Request request = new Request { Data = "Sample request data" };

                try
                {
                    Response response = await loadBalancer.ForwardRequestAsync(request);
                }
                catch (Exception ex)
                {
                }
                requestCounter++;
            }
        }
    }
}
