using Microsoft.AspNetCore.Mvc;

using Files = System.IO.File;

namespace LNU.NMMPH.API.Controllers
{
    [Route("api/files")]
    [ApiController]
    public class FilesController : ControllerBase
    {
        [HttpGet("{fileName}")]
        public IActionResult Get(string fileName)
        {
            string baseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            string fileDirectory = Path.Combine(baseDirectory, "Files");
            string filePath = Path.Combine(fileDirectory, fileName);

            if (!Files.Exists(filePath))
                return NotFound("File not found.");

            byte[] fileBytes = Files.ReadAllBytes(filePath);
  
            return File(fileBytes, "application/octet-stream", fileName);
        }
    }
}
