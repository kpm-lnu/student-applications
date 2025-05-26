"""
СТВОРЕННЯ СУРОГАТНОЇ RBF-МОДЕЛІ ДЛЯ (alpha, delta)
--------------------------------------------------
▪ вхід : alpha_delta_lhs_10.csv  ─ 10 (або скільки є) пар (alpha, delta)
▪ вихід: alpha_delta_results10.csv з колонкою max_prey
▪ вихід: 3-D графік поверхні  max_prey = f(alpha, delta)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401  (лише щоб не лаявся linter)

# === 1. Завантажуємо точки LHS =============================================
df = pd.read_csv("alpha_delta_lhs_20.csv")      # ← ваш CSV із LHS-точками
X  = df[["alpha", "delta"]].values              # тепер delta замість gamma

# === 2. Параметри моделі Лотки–Вольтерра ====================================
beta  = 0.1         # коеф. зустрічей (x*y) → зменшення жертв
gamma = 0.1         # смертність хижаків            (фіксовано)
t0, t_end = 0, 20
dt  = 0.1
t_eval = np.arange(t0, t_end + dt, dt)

def lotka_volterra(t, z, alpha, delta):
    """Система LV c параметрами (alpha, beta_const, delta, gamma_const)."""
    x, y = z
    dxdt = alpha * x - beta  * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# === 3. Розв’язуємо ОДУ для кожної (alpha, delta) ===========================
max_prey_list = []
for alpha, delta_i in X:                      # delta_i, щоб не плутати з глобал.
    sol = solve_ivp(
        lotka_volterra, [t0, t_end], [10, 5],
        t_eval=t_eval, args=(alpha, delta_i), method="RK45"
    )
    max_prey_list.append(np.max(sol.y[0]))

y = np.array(max_prey_list)

# === 4. Зберігаємо результати ===============================================
df["max_prey"] = y
df.to_csv("alpha_delta_results20.csv", index=False)
print("✅ Файл 'alpha_delta_results10.csv' збережено.")

# === 5. Будуємо RBF-інтерполятор ============================================
rbf_model = RBFInterpolator(X, y)   # kernel='multiquadric' (дефолт)

# === 6. Відмальовуємо поверхню ==============================================
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

alpha_range = np.linspace(0.3, 1.5, 50)
delta_range = np.linspace(0.001, 0.1, 50)
alpha_grid, delta_grid = np.meshgrid(alpha_range, delta_range)

z_grid = rbf_model(
    np.c_[alpha_grid.ravel(), delta_grid.ravel()]
).reshape(alpha_grid.shape)

ax.plot_surface(alpha_grid, delta_grid, z_grid, cmap="viridis", edgecolor="none")

ax.set_xlabel("alpha", fontsize=12)
ax.set_ylabel("delta", fontsize=12)
ax.set_zlabel("max_prey", fontsize=12)
ax.set_title("Сурогатна модель: max_prey по alpha і delta", fontsize=14)

plt.tight_layout()
plt.show()
