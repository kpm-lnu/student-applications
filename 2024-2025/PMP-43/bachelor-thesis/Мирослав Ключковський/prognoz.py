import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


class ModelConfig:
    def __init__(self):
        self.total_population = 4100000
        self.Nx, self.Ny = 30, 30
        self.T = 360
        self.sigma = 1 / 5.2
        self.gamma = 1 / 11.5
        self.beta_base = 0.5
        self.diffusion_coef = 0.3
        self.vaccination_day = None
        self.vaccination_coverage = 0.0
        self.quarantine_day = None
        self.quarantine_beta_scale = 1.0
        self.quarantine_diffusion_scale = 1.0

    def time_varying_beta(self, day):
        if self.quarantine_day is not None and day >= self.quarantine_day:
            return self.beta_base * self.quarantine_beta_scale
        return self.beta_base

    def time_varying_diffusion(self, day):
        if self.quarantine_day is not None and day >= self.quarantine_day:
            return self.diffusion_coef * self.quarantine_diffusion_scale
        return self.diffusion_coef


class SEIRModel:
    def __init__(self, config):
        self.cfg = config
        cell_population = self.cfg.total_population / (self.cfg.Nx * self.cfg.Ny)
        self.S = np.full((self.cfg.Nx, self.cfg.Ny), cell_population)
        self.E = np.zeros_like(self.S)
        self.I = np.zeros_like(self.S)
        self.R = np.zeros_like(self.S)

        # Стартова інфекція у центрі
        x0, y0 = self.cfg.Nx // 2, self.cfg.Ny // 2
        self.I[x0, y0] = 100
        self.E[x0, y0] = 300
        self.S[x0, y0] -= 400

        self.history = []

    def laplacian(self, u):
        kernel = np.array([[0.05, 0.1, 0.05],
                           [0.1, -0.6, 0.1],
                           [0.05, 0.1, 0.05]])
        return convolve2d(u, kernel, mode='same', boundary='fill')

    def compute_derivatives(self, S, E, I, R, day):
        D = self.cfg.time_varying_diffusion(day)
        beta = self.cfg.time_varying_beta(day)
        N = np.clip(S + E + I + R, 1e-6, None)

        infection = beta * S * I / N

        dS = -infection + D * self.laplacian(S)
        dE = infection - self.cfg.sigma * E + D * self.laplacian(E)
        dI = self.cfg.sigma * E - self.cfg.gamma * I + D * self.laplacian(I)
        dR = self.cfg.gamma * I + D * self.laplacian(R)

        return dS, dE, dI, dR

    def apply_vaccination(self):
        if self.cfg.vaccination_day is not None and self.day == self.cfg.vaccination_day:
            num_vaccinated = self.S * self.cfg.vaccination_coverage
            self.S -= num_vaccinated
            self.R += num_vaccinated

    def run(self):
        for day in range(self.cfg.T):
            self.day = day
            self.apply_vaccination()

            S, E, I, R = self.S.copy(), self.E.copy(), self.I.copy(), self.R.copy()

            k1 = self.compute_derivatives(S, E, I, R, day)
            k2 = self.compute_derivatives(*(S + 0.5 * k for k, S in zip(k1, [S, E, I, R])), day + 0.5)
            k3 = self.compute_derivatives(*(S + 0.5 * k for k, S in zip(k2, [S, E, I, R])), day + 0.5)
            k4 = self.compute_derivatives(*(S + k for k, S in zip(k3, [S, E, I, R])), day + 1)

            self.S += (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            self.E += (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            self.I += (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
            self.R += (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6

            self.S = np.clip(self.S, 0, None)
            self.E = np.clip(self.E, 0, None)
            self.I = np.clip(self.I, 0, None)
            self.R = np.clip(self.R, 0, None)

            totals = [np.sum(x) for x in [self.S, self.E, self.I, self.R]]
            self.history.append((day, *totals))

        return self.history


# Запуск 4 сценаріїв
def run_scenario(name, **kwargs):
    cfg = ModelConfig()
    for key, val in kwargs.items():
        setattr(cfg, key, val)
    model = SEIRModel(cfg)
    history = model.run()
    return history


base = run_scenario("base")
vaccination = run_scenario("vaccination", vaccination_day=30, vaccination_coverage=0.2)
quarantine = run_scenario("quarantine", quarantine_day=20, quarantine_beta_scale=0.5, quarantine_diffusion_scale=0.3)
both = run_scenario("both", vaccination_day=30, vaccination_coverage=0.2,
                    quarantine_day=20, quarantine_beta_scale=0.5, quarantine_diffusion_scale=0.3)

# Підготовка даних
def extract_data(histories, idx):
    return [[h[idx] / 1e6 for h in history] for history in histories]

histories = [base, vaccination, quarantine, both]
labels = ['Базовий', 'Вакцинація', 'Карантин', 'Вакцинація + Карантин']
S_all = extract_data(histories, 1)
E_all = extract_data(histories, 2)
I_all = extract_data(histories, 3)
R_all = extract_data(histories, 4)

# Побудова графіків
days = list(range(len(base)))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for i, (data, title) in enumerate(zip([S_all, E_all, I_all, R_all],
                                      ['Сприйнятливі (S)', 'Експоновані (E)', 'Інфіковані (I)', 'Одужалі (R)'])):
    ax = axs[i // 2, i % 2]
    for series, label in zip(data, labels):
        ax.plot(days, series, label=label, linewidth=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Дні')
    ax.set_ylabel('Кількість (млн)')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()
