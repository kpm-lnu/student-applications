import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Завантаження точок з LHS ===
df = pd.read_csv("train_alpha_beta_lhs_40.csv")
X = df[["alpha", "beta"]].values  # Вхідні параметри

# === Інші фіксовані параметри ===
delta = 0.01
gamma = 0.1
t0, t_end = 0, 20
dt = 0.1
t_eval = np.arange(t0, t_end + dt, dt)

# === Функція моделі Лотки–Вольтерра ===
def lotka_volterra(t, z, alpha, beta):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# === Обчислення max_prey для кожної пари (alpha, beta) ===
max_prey_list = []

for alpha, beta in X:
    sol = solve_ivp(lotka_volterra, [t0, t_end], [10, 5],
                    t_eval=t_eval, args=(alpha, beta), method='RK45')
    max_x = np.max(sol.y[0])
    max_prey_list.append(max_x)

y = np.array(max_prey_list)  # Це цільова змінна для сурогатної моделі

# === Збереження результатів у CSV ===
df['max_prey'] = y
df.to_csv("lotka_volterra_results40_alpha_beta.csv", index=False)

# === Побудова RBF-сурогатної моделі ===
rbf_model = RBFInterpolator(X, y)

# === Побудова 3D графіку ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Дані для 3D графіка
alpha_range = np.linspace(0.3, 1.5, 50)
beta_range = np.linspace(0.01, 0.22, 50)
alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)
z_grid = rbf_model(np.c_[alpha_grid.ravel(), beta_grid.ravel()]).reshape(alpha_grid.shape)

# Побудова поверхні
ax.plot_surface(alpha_grid, beta_grid, z_grid, cmap='viridis', edgecolor='none')

# Оформлення графіка
ax.set_xlabel('alpha', fontsize=12)
ax.set_ylabel('beta', fontsize=12)
ax.set_zlabel('max_prey', fontsize=12) #жертви
ax.set_title('Сурогатна модель: max_prey по alpha і beta', fontsize=14)
plt.tight_layout()

# Показ графіка
plt.show()