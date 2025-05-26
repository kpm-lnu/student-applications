using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Doploma4_Last_Edition
{
    public class InfectionData
    {
        public double Susceptible { get; set; } // susceptible
        public double Exposed { get; set; }   // exposed
        public double Infectious { get; set; }    // infectious
        public double Hospitalized { get; set; }   // hospitalized
        public double Critical { get; set; }    // critical
        public double Recovered { get; set; }   // recovered
        public double Deceased { get; set; }    // deceased
        public string Full => $"вразливі {(int)Susceptible} інкубаторний{(int)Exposed} хворі{(int)Infectious} вилікувані{(int)Recovered} смерті{(int)Deceased}";
    }
}
