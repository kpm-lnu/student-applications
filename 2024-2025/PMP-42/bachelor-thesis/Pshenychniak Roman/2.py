import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Завантаження даних
merged = pd.read_csv('input/SEIRVD.csv')

t = np.arange(len(merged))
N = 41_000_000
length = len(t)

S_data = merged['susceptible'].values
E_data = merged['exposed'].values
I_data = merged['infected'].values
R_data = merged['recovered'].values
V_data = merged['people_vaccinated'].values
D_data = merged['deaths_x'].values

# Модель SEIRVD
def seirvd_model_conditional_vacc(y, t, beta, sigma, gamma, mu, v_rate):
    S, E, I, R, V, D = y
    day = int(t)
    v = v_rate * S if day >= 340 else 0
    dSdt = -beta * S * I / N - v
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I
    dVdt = v
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dVdt, dDdt]

# Фіксовані параметри
fixed_sigma = 1/5.2     
fixed_gamma = 1/10      

# Функція втрат
def loss_reduced(params):
    beta, mu, v_rate, E0 = params
    y0 = [S_data[0], E0, I_data[0], R_data[0], V_data[0], D_data[0]]
    sol = odeint(
        seirvd_model_conditional_vacc,
        y0, t,
        args=(beta, fixed_sigma, fixed_gamma, mu, v_rate)
    )
    I_pred = sol[:length, 2]
    R_pred = sol[:length, 3]
    V_pred = sol[:length, 4]
    D_pred = sol[:length, 5]
    return (
        3.0 * np.mean((I_data - I_pred) ** 2) +
        1.5 * np.mean((R_data - R_pred) ** 2) +
        1.0 * np.mean((V_data - V_pred) ** 2) +
        1.0 * np.mean((D_data - D_pred) ** 2)
    )

# Початкові значення та межі
initial_params = [0.3, 0.005, 0.0005, 2000]
bounds = [(0.05, 0.6), (0.001, 0.05), (0.00000000005, 0.05), (0, 50000)]

# Оптимізація
result = minimize(loss_reduced, initial_params, bounds=bounds)
opt_beta, opt_mu, opt_vrate, opt_E0 = result.x

# Вивід оптимальних параметрів
print("Оптимальні параметри:")
print(f"beta (швидкість інфекції): {opt_beta:.6f}")
print(f"mu (швидкість смертності): {opt_mu:.6f}")
print(f"v_rate (швидкість вакцинації): {opt_vrate:.6f}")
print(f"E0 (початкова кількість експонованих): {opt_E0:.0f}")
print(f"Значення функції втрат (MSE): {result.fun:.6f}")
print(f"Статус оптимізації: {result.message}")

# Моделювання з оптимальними параметрами
y0_opt = [S_data[0], opt_E0, I_data[0], R_data[0], V_data[0], D_data[0]]
solution = odeint(
    seirvd_model_conditional_vacc,
    y0_opt, t,
    args=(opt_beta, fixed_sigma, fixed_gamma, opt_mu, opt_vrate)
)
S, E, I, R, V, D = solution[:length].T

# Візуалізація
plt.figure(figsize=(14, 8))
plt.plot(t, I_data, 'o', label='Реальні інфіковані')
plt.plot(t, R_data, 'o', label='Реальні одужалі')
plt.plot(t, V_data, 'o', label='Реальні вакциновані')
plt.plot(t, D_data, 'o', label='Реальні померлі')

plt.plot(t, I, '-', label='Модель: інфіковані')
plt.plot(t, R, '-', label='Модель: одужалі')
plt.plot(t, V, '-', label='Модель: вакциновані')
plt.plot(t, D, '-', label='Модель: померлі')

plt.xlabel('Дні з початку спостереження')
plt.ylabel('Кількість людей')
plt.title('SEIRVD модель з вакцинацією після 340-го дня')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()