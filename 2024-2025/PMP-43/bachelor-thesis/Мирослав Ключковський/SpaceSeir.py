# Stochastic SEIR Model with Spatial Diffusion and Uncertainty
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from scipy.signal import convolve2d
from io import StringIO

def download_ukraine_covid_data():
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    try:
        response = requests.get(url, timeout=15)
        df = pd.read_csv(StringIO(response.text))
    except Exception as e:
        print(f"Error: {e}, using local fallback")
        df = pd.read_csv(url)

    ukr_df = df[df['iso_code'] == 'UKR'].copy()
    ukr_df['date'] = pd.to_datetime(ukr_df['date'])
    ukr_df['new_cases'] = ukr_df['new_cases'].clip(lower=0).fillna(0).rolling(14, center=True).mean().fillna(0)
    ukr_df['total_cases'] = ukr_df['total_cases'].ffill().bfill()
    return ukr_df[['date', 'new_cases', 'total_cases', 'people_vaccinated_per_hundred']]

class ModelConfig:
    def __init__(self):
        self.total_population = 41e6
        self.Nx, self.Ny = 50, 50
        self.T = 180
        self.Nt = self.T
        self.simulation_start = datetime(2020, 2, 15)  # було 2020-03-15

        self.beta_scale = 1.0
        self.underreporting = 1.2
        self.sigma = 1 / 5.5
        self.gamma = 1 / 10.5
        self.diffusion_coef = 0.4
        self.vaccine_effect = 0.65
        self.real_vax = None

# ========== SEIR Model ==========
class SEIRModel:
    def __init__(self, config):
        self.cfg = config
        self.S = np.full((self.cfg.Nx, self.cfg.Ny), self.cfg.total_population / (self.cfg.Nx * self.cfg.Ny))
        self.E = np.zeros_like(self.S)
        self.I = np.zeros_like(self.S)
        self.R = np.zeros_like(self.S)
        self.V = np.zeros_like(self.S)

        # Локалізоване початкове зараження в центрі
        mid_x, mid_y = self.cfg.Nx // 2, self.cfg.Ny // 2
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                self.I[mid_x + dx, mid_y + dy] += 5
                self.S[mid_x + dx, mid_y + dy] -= 5

    def laplacian(self, u):
        kernel = np.array([[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]])
        return convolve2d(u, kernel, mode='same', boundary='fill')

    def time_varying_beta(self, day):
        base_beta = {0: 0.28, 30: 0.26, 60: 0.23, 90: 0.22, 120: 0.24, 150: 0.25}
        raw_beta = np.interp(day, list(base_beta.keys()), list(base_beta.values()))
        active_ratio = np.sum(self.I) / self.cfg.total_population
        return raw_beta * (1 - min(0.3, active_ratio * 3))

    def compute_derivatives(self, S, E, I, R, day, beta, sigma, gamma, D):
        dS_diff = D * 0.1 * self.laplacian(S)
        dE_diff = D * 0.05 * self.laplacian(E)
        dI_diff = D * 0.01 * self.laplacian(I)
        dR_diff = D * 0.1 * self.laplacian(R)

        N = np.clip(S + E + I + R, 1e-5, None)
        infection = beta * S * I / N

        dS = dS_diff - infection
        dE = dE_diff + infection - sigma * E
        dI = dI_diff + sigma * E - gamma * I
        dR = dR_diff + gamma * I

        return dS, dE, dI, dR

    def run(self):
        results = []
        D, sigma, gamma = self.cfg.diffusion_coef, self.cfg.sigma, self.cfg.gamma

        for day in range(self.cfg.T):
            if self.cfg.real_vax and day in self.cfg.real_vax:
                nu = self.cfg.real_vax[day] * self.cfg.vaccine_effect
                new_v = nu * self.S
                self.V += new_v
                self.S -= new_v

            noise = np.random.normal(0, 0.05)
            beta = max(0, self.time_varying_beta(day) * self.cfg.beta_scale * (1 + noise))

            S, E, I, R = self.S.copy(), self.E.copy(), self.I.copy(), self.R.copy()
            k1 = self.compute_derivatives(S, E, I, R, day, beta, sigma, gamma, D)
            k2 = self.compute_derivatives(*(S + 0.5 * k for k, S in zip(k1, [S, E, I, R])), day + 0.5, beta, sigma, gamma, D)
            k3 = self.compute_derivatives(*(S + 0.5 * k for k, S in zip(k2, [S, E, I, R])), day + 0.5, beta, sigma, gamma, D)
            k4 = self.compute_derivatives(*(S + k for k, S in zip(k3, [S, E, I, R])), day + 1, beta, sigma, gamma, D)

            self.S = np.clip(S + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6, 0, None)
            self.E = np.clip(E + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6, 0, None)
            self.I = np.clip(I + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6, 0, None)
            self.R = np.clip(R + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6, 0, None)

            new_cases = np.sum(sigma * self.E) * self.cfg.underreporting
            results.append([day, new_cases])

        return pd.DataFrame(results, columns=["Day", "New_Cases"])

def run_stochastic_simulations(cfg, runs=50):
    all_results = []
    for _ in range(runs):
        model = SEIRModel(cfg)
        sim = model.run()
        all_results.append(sim['New_Cases'].values)

    all_results = np.array(all_results)
    mean = np.mean(all_results, axis=0)
    lower = np.percentile(all_results, 5, axis=0)
    upper = np.percentile(all_results, 95, axis=0)

    return pd.DataFrame({
        'Day': np.arange(cfg.T),
        'Mean': mean,
        'Lower': lower,
        'Upper': upper
    })

def plot_with_uncertainty(sim_df, real_data):
    plt.figure(figsize=(14, 7))
    start_date = cfg.simulation_start
    dates = [start_date + timedelta(days=int(d)) for d in sim_df['Day']]
    real = real_data[(real_data['date'] >= dates[0]) & (real_data['date'] <= dates[-1])]

    plt.plot(dates, sim_df['Mean'], label='Середнє')
    plt.fill_between(dates, sim_df['Lower'], sim_df['Upper'], color='blue', alpha=0.2, label='90% інтервал')
    plt.plot(real['date'], real['new_cases'], label='Реальні дані', alpha=0.6)
    plt.title("Симуляція з стохастичністю")
    plt.ylabel("Нові випадки")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

# ========== Execution ==========
if __name__ == "__main__":
    real_data = download_ukraine_covid_data()
    cfg = ModelConfig()
    real_data['day'] = (real_data['date'] - cfg.simulation_start).dt.days
    vax_map = real_data[real_data['day'] <= cfg.T]['people_vaccinated_per_hundred'].fillna(0) / 100
    cfg.real_vax = vax_map.to_dict()

    sim_df = run_stochastic_simulations(cfg, runs=50)
    plot_with_uncertainty(sim_df, real_data)
