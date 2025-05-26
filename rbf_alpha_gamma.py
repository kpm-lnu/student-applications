"""
СТВОРЕННЯ СУРОГАТНОЇ RBF-МОДЕЛІ ДЛЯ (alpha, gamma)
--------------------------------------------------
▪ вхід:  alpha_gamma_lhs_20.csv  ─ 20 пар (alpha, gamma)
▪ вихід: alpha_gamma_results20.csv з колонкою max_prey
▪ вихід: 3-D графік поверхні max_prey = f(alpha, gamma)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === 1. Завантажуємо точки LHS =============================================
df = pd.read_csv("alpha_gamma_lhs_10.csv")
X = df[["alpha", "gamma"]].values  # ← тепер gamma замість beta

# === 2. Параметри моделі Лотки–Вольтерра ====================================
delta = 0.01           # фіксоване
beta  = 0.1            # ***Тепер beta фіксуємо, gamma варіюється***
t0, t_end = 0, 20
dt = 0.1
t_eval = np.arange(t0, t_end + dt, dt)

def lotka_volterra(t, z, alpha, gamma):
    """Система LV з параметрами (alpha, beta_const, gamma)."""
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# === 3. Розв’язуємо ОДУ для кожної (alpha, gamma) ===========================
max_prey_list = []
for alpha, gamma in X:
    sol = solve_ivp(
        lotka_volterra, [t0, t_end], [10, 5],
        t_eval=t_eval, args=(alpha, gamma), method="RK45"
    )
    max_prey_list.append(np.max(sol.y[0]))

y = np.array(max_prey_list)

# === 4. Зберігаємо результати ===============================================
df["max_prey"] = y
df.to_csv("alpha_gamma_results10.csv", index=False)
print("✅ Файл 'alpha_gamma_results10.csv' збережено.")

# === 5. Будуємо RBF-інтерполятор ============================================
rbf_model = RBFInterpolator(X, y)           # kernel='multiquadric' за замовч.

# === 6. Відмальовуємо поверхню ==============================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

alpha_range  = np.linspace(0.3,  1.5, 50)
gamma_range  = np.linspace(0.1,  0.35, 50)
alpha_grid, gamma_grid = np.meshgrid(alpha_range, gamma_range)

z_grid = rbf_model(
    np.c_[alpha_grid.ravel(), gamma_grid.ravel()]
).reshape(alpha_grid.shape)

surf = ax.plot_surface(
    alpha_grid, gamma_grid, z_grid,
    cmap="viridis", edgecolor="none"
)

ax.set_xlabel("alpha", fontsize=12)
ax.set_ylabel("gamma", fontsize=12)
ax.set_zlabel("max_prey", fontsize=12)
ax.set_title("Сурогатна модель: max_prey по alpha і gamma", fontsize=14)

plt.tight_layout()
plt.show()
