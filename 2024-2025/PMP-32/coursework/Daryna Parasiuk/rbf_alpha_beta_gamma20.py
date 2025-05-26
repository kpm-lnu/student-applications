import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Завантаження точок з LHS ===
df = pd.read_csv("train_alpha_beta_gamma_lhs_20.csv")
X = df[["alpha", "beta", "gamma"]].values  # Вхідні параметри

# === Інші фіксовані параметри ===
delta = 0.01
t0, t_end = 0, 20
dt = 0.1
t_eval = np.arange(t0, t_end + dt, dt)

# === Функція моделі Лотки–Вольтерра ===
def lotka_volterra(t, z, alpha, beta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# === Обчислення max_prey для кожної трійки (alpha, beta, gamma) ===
max_prey_list = []

for alpha, beta, gamma in X:
    sol = solve_ivp(lotka_volterra, [t0, t_end], [10, 5],
                    t_eval=t_eval, args=(alpha, beta, gamma), method='RK45')
    max_x = np.max(sol.y[0])
    max_prey_list.append(max_x)

y = np.array(max_prey_list)  # Це цільова змінна для сурогатної моделі

# === Збереження результатів у CSV ===
df['max_prey'] = y
df.to_csv("lotka_volterra_results20_alpha_beta_gamma.csv", index=False)

# rbf_model = RBFInterpolator(X, y)
# rbf_model = RBFInterpolator(X, y, kernel='gaussian', epsilon=0.05)
rbf_model = RBFInterpolator(X, y, kernel='cubic')
