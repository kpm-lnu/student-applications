"""
ПЕРЕВІРКА ЯКОСТІ RBF-МОДЕЛІ ДЛЯ (alpha, delta)
▪ Модель береться з alpha_delta_results10.csv  (колонки: alpha, delta, max_prey)
▪ Тестові точки — з test_alpha_delta_lhs_10.csv (alpha, delta)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# === 1. Дані для навчання ====================================================
df_train = pd.read_csv("alpha_delta_results40.csv")
required = {"alpha", "delta", "max_prey"}
if not required.issubset(df_train.columns):
    raise ValueError("У файлі мають бути стовпці: alpha, delta, max_prey")

X_train = df_train[["alpha", "delta"]].values
y_train = df_train["max_prey"].values

# === 2. Навчання RBF-моделі (відтворюємо) ===================================
rbf_model = RBFInterpolator(X_train, y_train)

# === 3. Тестова вибірка =====================================================
df_test = pd.read_csv("test_alpha_delta_lhs_40.csv")
if not {"alpha", "delta"}.issubset(df_test.columns):
    raise ValueError("У файлі test_alpha_delta_lhs_10.csv мають бути стовпці alpha, delta")

X_test = df_test[["alpha", "delta"]].values

# === 4. Істинні max_prey через інтегрування LV ==============================
beta  = 0.1     # постійна швидкість зустрічі
gamma = 0.1     # смертність хижаків (константа)
t0, t_end = 0, 20
dt = 0.1
t_eval = np.arange(t0, t_end + dt, dt)

def lotka_volterra(t, z, alpha, delta_param):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta_param * x * y - gamma * y
    return [dxdt, dydt]

true_max_prey = []
for alpha, delta_param in X_test:
    sol = solve_ivp(
        lotka_volterra, [t0, t_end], [10, 5],
        t_eval=t_eval, args=(alpha, delta_param), method="RK45"
    )
    true_max_prey.append(np.max(sol.y[0]))
true_max_prey = np.array(true_max_prey)

# === 5. Прогноз моделі ======================================================
predicted_max_prey = rbf_model(X_test)

# === 6. Метрики точності ====================================================
mae  = mean_absolute_error(true_max_prey, predicted_max_prey)
rmse = np.sqrt(mean_squared_error(true_max_prey, predicted_max_prey))
r2   = r2_score(true_max_prey, predicted_max_prey)

print(f"MAE : {mae :.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2  :.4f}")

# === 7. Візуалізація ========================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    X_test[:, 0], X_test[:, 1], true_max_prey,
    c="red", label="Істинні max_prey", s=60
)
ax.scatter(
    X_test[:, 0], X_test[:, 1], predicted_max_prey,
    c="blue", marker="^", label="Прогноз RBF", s=60
)
ax.set_xlabel("alpha", fontsize=12)
ax.set_ylabel("delta", fontsize=12)
ax.set_zlabel("max_prey", fontsize=12)
ax.set_title("Справжні vs Прогнозовані значення (alpha, delta)", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
