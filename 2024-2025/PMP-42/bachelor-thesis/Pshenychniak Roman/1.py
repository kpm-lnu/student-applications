import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Завантаження даних
filtered_data = pd.read_csv('input/SEIR.csv')
filtered_data['date'] = pd.to_datetime(filtered_data['date'])

# Дані
I_data = filtered_data['infected'].values
R_data = filtered_data['recovered_deceased'].values
N = 41_000_000
S_data = N - I_data - R_data

# Часова шкала
t = (filtered_data['date'] - filtered_data['date'].iloc[0]).dt.days.values

# SEIR модель
def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Функція втрат
def loss(params):
    beta, sigma, gamma, E0 = params
    y0 = S_data[0], E0, I_data[0], R_data[0]
    sol = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
    I_pred, R_pred = sol[:, 2], sol[:, 3]
    w_I, w_R = 3.0, 1.0
    return w_I * np.mean((I_data - I_pred)**2) + w_R * np.mean((R_data - R_pred)**2)

# Оптимізація
bounds = [(0.1, 1.0), (0.15, 0.3), (0.05, 0.15), (100, 10000)]
initial_params = [0.1, 0.2997, 0.899, 1000]
result = differential_evolution(loss, bounds, maxiter=1000, popsize=15)
opt_beta, opt_sigma, opt_gamma, opt_E0 = result.x
print(f"Оптимальні параметри: beta={opt_beta:.4f}, sigma={opt_sigma:.4f}, gamma={opt_gamma:.4f}, E0={opt_E0:.0f}")

# Модельні значення
y0_opt = S_data[0], opt_E0, I_data[0], R_data[0]
solution = odeint(seir_model, y0_opt, t, args=(opt_beta, opt_sigma, opt_gamma))
S, E, I, R = solution.T

# Графіки
plt.figure(figsize=(12, 6))
plt.plot(t, I_data, 'o', label='Реальні інфіковані (active)')
plt.plot(t, I, '-', label='Модельні інфіковані')
plt.plot(t, R_data, 'o', label='Реальні одужалі + померлі')
plt.plot(t, R, '-', label='Модельні одужалі')
plt.xlabel('Дні з 21.03.2020')
plt.ylabel('Кількість людей')
plt.legend()
plt.title('SEIR модель')
plt.grid(True)
plt.show()

# # Графік помилок
# plt.figure(figsize=(12, 6))
# plt.plot(t, I_data - I, 'o-', label='Помилка для інфікованих')
# plt.axhline(0, color='black', linestyle='--')
# plt.xlabel('Дні')
# plt.ylabel('Помилка (I_data - I_pred)')
# plt.legend()
# plt.grid(True)
# plt.show()