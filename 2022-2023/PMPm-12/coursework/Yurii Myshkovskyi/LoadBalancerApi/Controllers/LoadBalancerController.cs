using Microsoft.AspNetCore.Mvc;

namespace LoadBalancerApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class LoadBalancerController : ControllerBase
    {
        private readonly ILoadBalancer _loadBalancer;

        public LoadBalancerController(ILoadBalancer loadBalancer)
        {
            _loadBalancer = loadBalancer;
        }

        [HttpPost]
        public async Task<IActionResult> ProcessRequestAsync([FromBody] Request request)
        {
            var response = await _loadBalancer.ForwardRequestAsync(request);
            return Ok(response);
        }
    }
}
