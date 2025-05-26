"""
ПЕРЕВІРКА ЯКОСТІ GAUSSIAN-RBF-МОДЕЛІ ДЛЯ (beta, gamma)
------------------------------------------------------
▪ Навчальна вибірка:  beta_gamma_gauss_results5.csv  (beta, gamma, max_prey)
▪ Тестові точки:      test_beta_gamma_lhs_5.csv      (beta, gamma)
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === 1. Дані для навчання (ті ж, що й у моделі) =============================
df_train = pd.read_csv("beta_gamma_cubic_results40.csv")
required_cols = {"beta", "gamma", "max_prey"}
if not required_cols.issubset(df_train.columns):
    raise ValueError("У файлі мають бути стовпці: beta, gamma, max_prey")

X_train = df_train[["beta", "gamma"]].values
y_train = df_train["max_prey"].values

# === 2. Відтворюємо Gaussian-RBF-модель =====================================
rbf_model = RBFInterpolator(X_train, y_train, kernel="gaussian", epsilon=0.05)

# === 2. Навчання (відтворюємо модель з кубічним ядром) =====================
rbf_model = RBFInterpolator(X_train, y_train, kernel="cubic")

# === 3. Тестова вибірка =====================================================
df_test = pd.read_csv("test_beta_gamma_lhs_40.csv")
if not {"beta", "gamma"}.issubset(df_test.columns):
    raise ValueError("У файлі test_beta_gamma_lhs_5.csv мають бути стовпці beta, gamma")

X_test = df_test[["beta", "gamma"]].values

# === 4. Істинні max_prey через інтегрування LV ==============================
alpha = 1.0      # фіксоване (той самий, що й у train-скрипті)
delta = 0.01     # фіксоване

t0, t_end = 0, 20
dt = 0.1
t_eval = np.arange(t0, t_end + dt, dt)

def lotka_volterra(t, z, beta_param, gamma_param):
    x, y = z
    dxdt = alpha * x - beta_param * x * y
    dydt = delta * x * y - gamma_param * y
    return [dxdt, dydt]

true_max_prey = []
for beta_param, gamma_param in X_test:
    sol = solve_ivp(
        lotka_volterra, [t0, t_end], [10, 5],
        t_eval=t_eval, args=(beta_param, gamma_param), method="RK45"
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
    c="blue", marker="^", label="Прогноз Gaussian-RBF", s=60
)
ax.set_xlabel("beta", fontsize=12)
ax.set_ylabel("gamma", fontsize=12)
ax.set_zlabel("max_prey", fontsize=12)
ax.set_title("Справжні vs Прогнозовані значення (beta, gamma)\nGaussian RBF", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
