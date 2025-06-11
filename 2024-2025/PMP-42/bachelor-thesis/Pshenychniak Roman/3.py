import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# === 1. Завантаження даних ===
filtered_data = pd.read_csv("input/SEIRS.csv")
filtered_data['date'] = pd.to_datetime(filtered_data['date'])

# === 2. Підготовка змінних ===
t = np.arange(len(filtered_data))  # часовий вектор
N = filtered_data['population'].iloc[0]

R_data = filtered_data['recovered'].values
D_data = filtered_data['deaths'].values
I_data = filtered_data['confirmed'].values

# Щоденні вилучені (одужання + смерті)
filtered_data['daily_removed'] = filtered_data['recovered'].diff().fillna(0).clip(lower=0) + \
                        filtered_data['deaths'].diff().fillna(0).clip(lower=0)
R_data_daily = filtered_data['daily_removed'].values

# === 3. SEIRS-модель ===
def seirs_model(y, t, beta, sigma, gamma, xi):
    S, E, I, R = y
    dSdt = -beta * S * I / N + xi * R
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - xi * R
    return [dSdt, dEdt, dIdt, dRdt]

# === 4. Функція втрат ===
def loss_daily(params):
    beta, sigma, gamma, xi, E0 = params
    S0 = N - I_data[0] - R_data[0] - E0
    y0 = [S0, E0, I_data[0], R_data[0]]
    sol = odeint(seirs_model, y0, t, args=(beta, sigma, gamma, xi))
    I_pred = sol[:, 2]
    R_model_daily = np.diff(sol[:, 3], prepend=sol[0, 3])  # дельта R
    return (
        3.0 * np.mean((I_data - I_pred) ** 2) +
        1.0 * np.mean((R_data_daily - R_model_daily) ** 2)
    )

# === 5. Параметри та оптимізація ===
initial_params = [0.4, 1/5.2, 1/10, 0.001, 1000]
bounds = [(0.0001, 1), (0.0001, 1), (0.0001, 1), (0, 1), (0, 100000)]

result = minimize(loss_daily, initial_params, bounds=bounds)
opt_beta, opt_sigma, opt_gamma, opt_xi, opt_E0 = result.x

# === 6. Вивід оптимальних параметрів ===
print("Оптимальні параметри:")
print(f"beta (швидкість інфекції): {opt_beta:.6f}")
print(f"sigma (швидкість переходу E -> I): {opt_sigma:.6f}")
print(f"gamma (швидкість одужання): {opt_gamma:.6f}")
print(f"xi (швидкість повернення R -> S): {opt_xi:.6f}")
print(f"E0 (початкова кількість експонованих): {opt_E0:.0f}")
print(f"Значення функції втрат (MSE): {result.fun:.6f}")
print(f"Статус оптимізації: {result.message}")

# === 7. Розв'язання системи ===
S0 = N - I_data[0] - R_data[0] - opt_E0
y0_opt = [S0, opt_E0, I_data[0], R_data[0]]
solution = odeint(seirs_model, y0_opt, t, args=(opt_beta, opt_sigma, opt_gamma, opt_xi))
S, E, I, R = solution.T
R_model_daily = np.diff(R, prepend=R[0])

# === 8. Побудова графіка ===
plt.figure(figsize=(14, 7))
plt.plot(t, I_data, 'o', label='Реальні інфіковані (активні)', markersize=3)
plt.plot(t, R_data_daily, 'o', label='Реальні щоденні вилучені (одужання + смерті)', markersize=3)

plt.plot(t, I, '-', label='Модель: інфіковані')
plt.plot(t, R_model_daily, '-', label='Модель: щоденні вилучені')

plt.xlabel("Дні з початку 2022 року")
plt.ylabel("Кількість людей")
plt.title("SEIRS модель")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()