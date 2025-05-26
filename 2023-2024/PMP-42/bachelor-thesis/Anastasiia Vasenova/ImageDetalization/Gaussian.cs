using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Diploma
{
    public class Gaussian
    {
        private double a = new(); // weight of noise
        private int x = new();
        private int y = new();

        public double A { get => a; set => a = value; }
        public int X { get => x; set => x = value; }
        public int Y { get => y; set => y = value; }

        public Gaussian(double a, int x, int y)
        {
            A = a;
            X = x;
            Y = y;
        }

        public override string ToString()
        {
            string res = "";
            res += A.ToString() + ",\t" + X.ToString() + ",\t" + Y.ToString() + "\n";
            return res;
        }
    }
}