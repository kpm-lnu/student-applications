using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Doploma4_Last_Edition
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            
        }

        private async void button1_Click(object sender, EventArgs e)
        {
            
            Modeling();

            
            
        }
        public void Modeling()
        {

            double S = double.Parse(PeopleCount.Text); // susceptible
            double E = double.Parse(InfectiousCount.Text);    // exposed
            double I = 0;    // infectious
            double H = 0;    // hospitalized
            double C = 0;    // critical
            double R = 0;    // recovered
            double D = 0;    // deceased

            // Parameters
            double beta = double.Parse(Beta.Text); // infection rate
            double alpha = double.Parse(Alpha.Text); // rate of progression from exposed to infectious
            double gamma = double.Parse(Gamma.Text); // recovery rate
            double mu = double.Parse(Mu.Text); // death rate
            double theta = double.Parse(Theta.Text); // hospitalization rate
            double phi = double.Parse(Phi.Text); // critical care rate
            double p = double.Parse(P.Text); // proportion of infectious cases that are not hospitalized
            double q = double.Parse(Q.Text); // proportion of hospitalized cases that become critical

            // Time parameters
            double dt = 0.001; // time step
            double t = 0;    // current time
            double tmax = 20; // maximum time
            List<InfectionData> infectionDatas = new List<InfectionData>();

            // Euler's method
            while (t < tmax)
            {
                // Calculate derivatives
                double dSdt = (-beta * S * (I + theta * H + phi * C));
                double dEdt = (beta * S * (I + theta * H + phi * C) - alpha * E);
                double dIdt = ((1 - p) * alpha * E - (gamma + mu) * I - theta * p * I - phi * q * I);
                double dHdt = (theta * p * I - (gamma + mu) * H);
                double dCdt = (phi * q * I - (gamma + mu) * C);
                double dRdt = (gamma * (I + H + C));
                double dDdt = (mu * (I + H + C));

                // Runge-Kutta integration
                double h = dt; // step size
                double t1 = t + h / 2;

                double k1_S = dSdt;
                double k1_E = dEdt;
                double k1_I = dIdt;
                double k1_H = dHdt;
                double k1_C = dCdt;
                double k1_R = dRdt;
                double k1_D = dDdt;

                double k2_S = (-beta * (S + h / 2 * k1_S) * (I + theta * (H + h / 2 * k1_H) + phi * (C + h / 2 * k1_C)));
                double k2_E = (beta * (S + h / 2 * k1_S) * (I + theta * (H + h / 2 * k1_H) + phi * (C + h / 2 * k1_C)) - alpha * (E + h / 2 * k1_E));
                double k2_I = ((1 - p) * alpha * (E + h / 2 * k1_E) - (gamma + mu) * (I + h / 2 * k1_I) - theta * p * (I + h / 2 * k1_I) - phi * q * (I + h / 2 * k1_I));
                double k2_H = (theta * p * (I + h / 2 * k1_I) - (gamma + mu) * (H + h / 2 * k1_H));
                double k2_C = (phi * q * (I + h / 2 * k1_I) - (gamma + mu) * (C + h / 2 * k1_C));
                double k2_R = (gamma * ((I + h / 2 * k1_I) + (H + h / 2 * k1_H) + (C + h / 2 * k1_C)));
                double k2_D = (mu * ((I + h / 2 * k1_I) + (H + h / 2 * k1_H) + (C + h / 2 * k1_C)));

                double k3_S = (-beta * (S + h / 2 * k2_S) * (I + theta * (H + h / 2 * k2_H) + phi * (C + h / 2 * k2_C)));
                double k3_E = (beta * (S + h / 2 * k2_S) * (I + theta * (H + h / 2 * k2_H) + phi * (C + h / 2 * k2_C)) - alpha * (E + h / 2 * k2_E));
                double k3_I = ((1 - p) * alpha * (E + h / 2 * k2_E) - (gamma + mu) * (I + h / 2 * k2_I) - theta * p * (I + h / 2 * k2_I) - phi * q * (I + h / 2 * k2_I));
                double k3_H = (theta * p * (I + h / 2 * k2_I) - (gamma + mu) * (H + h / 2 * k2_H));
                double k3_C = (phi * q * (I + h / 2 * k2_I) - (gamma + mu) * (C + h / 2 * k2_C));
                double k3_R = (gamma * ((I + h / 2 * k2_I) + (H + h / 2 * k2_H) + (C + h / 2 * k2_C)));
                double k3_D = (mu * ((I + h / 2 * k2_I) + (H + h / 2 * k2_H) + (C + h / 2 * k2_C)));

                double k4_S = (-beta * (S + h * k3_S) * (I + theta * (H + h * k3_H) + phi * (C + h * k3_C)));
                double k4_E = (beta * (S + h * k3_S) * (I + theta * (H + h * k3_H) + phi * (C + h * k3_C)) - alpha * (E + h * k3_E));
                double k4_I = ((1 - p) * alpha * (E + h * k3_E) - (gamma + mu) * (I + h * k3_I) - theta * p * (I + h * k3_I) - phi * q * (I + h * k3_I));
                double k4_H = (theta * p * (I + h * k3_I) - (gamma + mu) * (H + h * k3_H));
                double k4_C = (phi * q * (I + h * k3_I) - (gamma + mu) * (C + h * k3_C));
                double k4_R = (gamma * ((I + h * k3_I) + (H + h * k3_H) + (C + h * k3_C)));
                double k4_D = (mu * ((I + h * k3_I) + (H + h * k3_H) + (C + h * k3_C)));

                // Update variables
                S += (h / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S);
                E += (h / 6) * (k1_E + 2 * k2_E + 2 * k3_E + k4_E);
                I += (h / 6) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I);
                H += (h / 6) * (k1_H + 2 * k2_H + 2 * k3_H + k4_H);
                C += (h / 6) * (k1_C + 2 * k2_C + 2 * k3_C + k4_C);
                R += (h / 6) * (k1_R + 2 * k2_R + 2 * k3_R + k4_R);
                D += (h / 6) * (k1_D + 2 * k2_D + 2 * k3_D + k4_D);

                // Update time
                t += h;
                Chart1.Series["Susceptible"].Points.AddXY(t, S);
                Chart1.Series["Exposed"].Points.AddXY(t, E);
                Chart1.Series["Infection"].Points.AddXY(t, I);
                Chart1.Series["Hospitalized"].Points.AddXY(t, H);
                Chart1.Series["Critical"].Points.AddXY(t, C);
                Chart1.Series["Recovered"].Points.AddXY(t, R);
                Chart1.Series["Deceased"].Points.AddXY(t, D);
                // Add data point
                infectionDatas.Add(new InfectionData()
                {
                    Susceptible = S,
                    Exposed = E,
                    Infectious = I,
                    Hospitalized = H,
                    Critical = C,
                    Recovered = R,
                    Deceased = D,
                });
            }

            listBox1.DataSource = infectionDatas;
            listBox1.DisplayMember = "Full";

        }
        

        private void chart1_Click(object sender, EventArgs e)
        {

        }

        private void PeopleCount_TextChanged(object sender, EventArgs e)
        {

        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void textBox3_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
