namespace LNU.NMMPH.API.Models.Parameters
{
    public class PoissonMethodParameters
    {
        public int Nx { get; set; } = 10;
        public int Ny { get; set; } = 10;
        public double A { get; set; } = 1.0;
        public double B { get; set; } = 1.0;
        public double F { get; set; } = 1.0;
    }
}
