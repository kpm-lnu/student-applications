from scipy.interpolate import RBFInterpolator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.integrate import solve_ivp

# === 1. Завантаження даних для навчання ===
df_train = pd.read_csv("lotka_volterra_results40_alpha_beta_gamma.csv")  # <-- Файл для 3 змінних

# Перевірка, чи є потрібні стовпці
if not all(col in df_train.columns for col in ["alpha", "beta", "gamma", "max_prey"]):
    raise ValueError("У файлі мають бути стовпці: alpha, beta, gamma, max_prey")

X_train = df_train[["alpha", "beta", "gamma"]].values
y_train = df_train["max_prey"].values

# rbf_model = RBFInterpolator(X_train, y_train)
# rbf_model = RBFInterpolator(X_train, y_train, kernel='gaussian', epsilon=0.05)
rbf_model = RBFInterpolator(X_train, y_train, kernel='cubic')


# === 3. Завантаження тестової вибірки ===
df_test = pd.read_csv("test_alpha_beta_gamma_lhs_40.csv")  # <-- тестовий файл для 3 змінних
if not all(col in df_test.columns for col in ["alpha", "beta", "gamma"]):
    raise ValueError("У файлі мають бути стовпці: alpha, beta, gamma")

X_test = df_test[["alpha", "beta", "gamma"]].values

# === 4. Обчислення істинних max_prey ===
delta = 0.01  # delta поки фіксоване
t0, t_end = 0, 20
dt = 0.1
t_eval = np.arange(t0, t_end + dt, dt)

def lotka_volterra(t, z, alpha, beta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

true_max_prey = []
for alpha, beta, gamma in X_test:
    sol = solve_ivp(lotka_volterra, [t0, t_end], [10, 5],
                    t_eval=t_eval, args=(alpha, beta, gamma), method='RK45')
    true_max_prey.append(np.max(sol.y[0]))

true_max_prey = np.array(true_max_prey)

# === 5. Прогноз сурогатної моделі ===
predicted_max_prey = rbf_model(X_test)

# === 6. MAE ===
mae = mean_absolute_error(true_max_prey, predicted_max_prey)
print(f"Середня абсолютна похибка (MAE): {mae:.4f}")

# === 7. RMSE ===
rmse = np.sqrt(mean_squared_error(true_max_prey, predicted_max_prey))
print(f"Корінь середньоквадратичної похибки (RMSE): {rmse:.4f}")

# === 8. R²
r2 = r2_score(true_max_prey, predicted_max_prey)
print(f"R²: {r2:.4f}")

# === 9. Побудова графіка ===
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], true_max_prey, c='red', label='Істинні max_prey', s=50)
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], predicted_max_prey, c='blue', marker='^', label='Прогноз моделі', s=50)
ax.set_xlabel('alpha', fontsize=12)
ax.set_ylabel('beta', fontsize=12)
ax.set_zlabel('max_prey', fontsize=12)
ax.set_title('Справжні vs Прогнозовані значення (3 змінні)', fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
