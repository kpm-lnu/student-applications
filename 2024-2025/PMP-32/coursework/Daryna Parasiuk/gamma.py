import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

# === Інші фіксовані параметри ===
alpha = 1
delta = 0.01
beta = 0.1
t0, t_end = 0, 20
dt = 0.1
t_eval = np.arange(t0, t_end + dt, dt)

# === Функція моделі Лотки–Вольтерра ===
def lotka_volterra(t, z, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# === Діапазон gamma для побудови кривої ===
gamma_range = np.linspace(0.1, 0.35, 200).reshape(-1, 1)
# === Кількість точок для побудови ===
n_points_list = [5, 10, 20, 40]

plt.figure(figsize=(12, 10))

for i, n_points in enumerate(n_points_list, 1):
    # === Завантаження TRAIN і TEST точок ===
    file_name_train = f"train_gamma_lhs_{n_points}.csv"
    df_train = pd.read_csv(file_name_train)
    X_train = df_train[["gamma"]].values  # Використовуємо правильну назву стовпця "gamma"

    file_name_test = f"test_gamma_lhs_{n_points}.csv"
    df_test = pd.read_csv(file_name_test)
    X_test = df_test[["gamma"]].values  # Використовуємо правильну назву стовпця "gamma"

    # === Розрахунок max_prey для TRAIN ===
    max_prey_train = []
    for (gamma,) in X_train:
        sol = solve_ivp(lotka_volterra, [t0, t_end], [10, 5], t_eval=t_eval, args=(gamma,))
        max_x = np.max(sol.y[0])
        max_prey_train.append(max_x)

    # === RBF-сурогатна модель ===
    rbf_model = RBFInterpolator(X_train, max_prey_train)

    # === Прогноз для TEST точок ===
    max_prey_test = rbf_model(X_test)

    # === Побудова кривої ===
    max_prey_curve = rbf_model(gamma_range)

    # === Побудова графіка ===
    plt.subplot(2, 2, i)
    plt.plot(gamma_range, max_prey_curve, color='green', label='Сурогатна модель')
    plt.scatter(X_train, max_prey_train, color='red', label='Навчальні точки', s=40)
    plt.scatter(X_test, max_prey_test, color='blue', label='Прогнозовані тестові точки', s=40)
    plt.xlabel('Gamma')
    plt.ylabel('Max Prey')
    plt.title(f'Навчальні точки: {n_points}')
    plt.grid(True)
    plt.legend()

    # === Збереження результатів у CSV ===
    results_df = pd.DataFrame({
        'gamma_train': X_train.flatten(),
        'max_prey_train': max_prey_train,
        'gamma_test': X_test.flatten(),
        'max_prey_test': max_prey_test
    })

    results_df.to_csv(f"lotka_volterra_results_gamma{n_points}.csv", index=False)

plt.suptitle('RBF-інтерполяція для різної кількості навчальних точок (gamma)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

