from scipy.interpolate import RBFInterpolator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.integrate import solve_ivp

# === 1. Параметри ===
alpha = 1.0  # фіксоване
delta = 0.01
beta = 0.1  # фіксоване значення beta
t0, t_end = 0, 20
dt = 0.1
t_eval = np.arange(t0, t_end + dt, dt)

# === Функція моделі Лотки–Вольтерра ===
def lotka_volterra(t, z, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y  # фіксоване beta
    dydt = delta * x * y - gamma * y  # змінна gamma
    return [dxdt, dydt]

# === 2. Обробка для різної кількості точок ===
n_points_list = [5, 10, 20, 40]
plt.figure(figsize=(12, 10))

for i, n_points in enumerate(n_points_list, 1):
    # Завантаження тестової вибірки для кожної кількості точок
    test_file_name = f"test_gamma_lhs_{n_points}.csv"
    df_test = pd.read_csv(test_file_name)

    # Перевірка на наявність стовпця 'gamma'
    if "gamma" not in df_test.columns:
        raise ValueError(f"У тестовому файлі {test_file_name} має бути стовпець 'gamma'")

    X_test = df_test[["gamma"]].values

    # Обчислення істинних значень max_prey через модель
    true_max_prey = []
    for (gamma,) in X_test:
        sol = solve_ivp(lotka_volterra, [t0, t_end], [10, 5], t_eval=t_eval, args=(gamma,), method='RK45')
        true_max_prey.append(np.max(sol.y[0]))

    true_max_prey = np.array(true_max_prey)

    # Завантаження навчальної вибірки для кожної кількості точок
    file_name_train = f"lotka_volterra_results_gamma{n_points}.csv"
    df_train = pd.read_csv(file_name_train)

    # Оновлення на правильні стовпці
    if not all(col in df_train.columns for col in ["gamma_train", "max_prey_train"]):
        raise ValueError(f"У файлі {file_name_train} мають бути стовпці: gamma_train, max_prey_train")

    X_train = df_train[["gamma_train"]].values
    y_train = df_train["max_prey_train"].values

    # Навчання RBF-моделі
    # rbf_model = RBFInterpolator(X_train, y_train)
    # rbf_model = RBFInterpolator(X_train, y_train, kernel='gaussian', epsilon=0.05)
    rbf_model = RBFInterpolator(X_train, y_train, kernel='cubic')

    # Прогноз моделі
    predicted_max_prey = rbf_model(X_test)

    # Метрики якості
    mae = mean_absolute_error(true_max_prey, predicted_max_prey)
    rmse = np.sqrt(mean_squared_error(true_max_prey, predicted_max_prey))
    r2 = r2_score(true_max_prey, predicted_max_prey)

    print(f"Для {n_points} точок:")
    print(f"Середня абсолютна похибка (MAE): {mae:.4f}")
    print(f"Корінь середньоквадратичної похибки (RMSE): {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print("-" * 40)

    # Побудова графіка
    plt.subplot(2, 2, i)
    plt.plot(X_test, true_max_prey, 'ro', label='Істинні max_prey')
    plt.plot(X_test, predicted_max_prey, 'b^', label='Прогноз моделі')
    plt.xlabel('gamma', fontsize=12)
    plt.ylabel('max_prey', fontsize=12)
    plt.title(f'RBF: Істинні vs Прогнозовані значення ({n_points} точок)', fontsize=14)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()