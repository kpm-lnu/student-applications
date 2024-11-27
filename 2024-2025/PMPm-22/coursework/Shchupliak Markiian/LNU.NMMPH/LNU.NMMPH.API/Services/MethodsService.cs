using LNU.NMMPH.API.Interface;
using LNU.NMMPH.API.Interface.Methods;
using LNU.NMMPH.API.Models;
using LNU.NMMPH.API.Services.Methods;

namespace LNU.NMMPH.API.Services
{
    public class MethodsService : IMethodsService
    {
        private readonly IEulerMethod _eulerMethod;
        private readonly IRungeKuttaMethod _rungeKuttaMethod;

        public MethodsService(IEulerMethod eulerMethod, IRungeKuttaMethod rungeKuttaMethod)
        {
            _eulerMethod = eulerMethod;
            _rungeKuttaMethod = rungeKuttaMethod;
        }

        public async Task<double> Execute(Method method, IFormFile file)
        {
            string fileString = await ConvertIFormFileToString(file);

            return method switch
            {
                Method.EulerMethod => await _eulerMethod.ExecuteStudent(fileString),
                Method.RungeKuttaMethod => await _rungeKuttaMethod.ExecuteStudent(fileString),
                _ => throw new NotImplementedException(),
            };
        }

        private async Task<string> ConvertIFormFileToString(IFormFile file)
        {
            using var reader = new StreamReader(file.OpenReadStream());
            string fileContent = await reader.ReadToEndAsync();

            return fileContent;
        }

        private async Task<string> ReadFileFromIFormFile(IFormFile file)
        {
            // Define the path where the temporary file will be saved
            var tempFilePath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + Path.GetExtension(file.FileName));

            // Save the IFormFile to the temporary file
            using (var stream = new FileStream(tempFilePath, FileMode.Create))
            {
                await file.CopyToAsync(stream);
            }

            // Now, use File.ReadAllText to read the contents of the file
            string fileContent = File.ReadAllText(tempFilePath);

            // Optionally, delete the temporary file if you no longer need it
            File.Delete(tempFilePath);

            return fileContent;
        }
    }
}
