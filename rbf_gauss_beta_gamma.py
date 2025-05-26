"""
СУРОГАТНА RBF-МОДЕЛЬ ДЛЯ (beta, gamma) З ГАУСОВИМ ЯДРОМ
-------------------------------------------------------
▪ вхід : beta_gamma_lhs_5.csv          ─ LHS-точки (beta, gamma)
▪ вихід: beta_gamma_gauss_results5.csv ─ з колонкою max_prey
▪ вихід: 3-D графік max_prey = f(beta, gamma)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === 1. Завантажуємо точки LHS =============================================
df = pd.read_csv("beta_gamma_lhs_40.csv")
X  = df[["beta", "gamma"]].values

# === 2. Фіксовані параметри Лотки–Вольтерра =================================
alpha = 1.0      # зростання жертв
delta = 0.01     # конверсія жертв у хижаків

t0, t_end = 0, 20
dt        = 0.1
t_eval    = np.arange(t0, t_end + dt, dt)

def lotka_volterra(t, z, beta_p, gamma_p):
    x, y = z
    dxdt = alpha * x - beta_p * x * y
    dydt = delta * x * y - gamma_p * y
    return [dxdt, dydt]

# === 3. Обчислюємо max_prey для кожної (beta, gamma) ========================
max_prey_list = []
for beta_i, gamma_i in X:
    sol = solve_ivp(
        lotka_volterra, [t0, t_end], [10, 5],
        t_eval=t_eval, args=(beta_i, gamma_i), method="RK45"
    )
    max_prey_list.append(np.max(sol.y[0]))

y = np.array(max_prey_list)

# === 4. Зберігаємо результати ===============================================
df["max_prey"] = y
df.to_csv("beta_gamma_cubic_results40.csv", index=False)
print("✅ Файл 'beta_gamma_gauss_results5.csv' збережено.")

# === 5. Будуємо RBF-інтерполятор з ГАУСОВИМ ядром ============================
# Виправлено: додаємо epsilon
# rbf_model = RBFInterpolator(X, y, kernel="gaussian", epsilon=0.05)

# === 5. Будуємо RBF-інтерполятор з кубічним ядром ============================
rbf_model = RBFInterpolator(X, y, kernel="cubic")

# === 6. Відмальовуємо поверхню ==============================================
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

beta_range  = np.linspace(0.025, 0.25, 60)
gamma_range = np.linspace(0.1,   0.35, 60)
beta_grid, gamma_grid = np.meshgrid(beta_range, gamma_range)

z_grid = rbf_model(
    np.c_[beta_grid.ravel(), gamma_grid.ravel()]
).reshape(beta_grid.shape)

ax.plot_surface(beta_grid, gamma_grid, z_grid, cmap="viridis", edgecolor="none")

ax.set_xlabel("beta", fontsize=12)
ax.set_ylabel("gamma", fontsize=12)
ax.set_zlabel("max_prey", fontsize=12)
ax.set_title("Сурогатна модель (Gaussian RBF): max_prey по beta і gamma", fontsize=14)

plt.tight_layout()
plt.show()
