import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# === 1. Завантаження даних ===
filtered_data = pd.read_csv('input/SEIRS.csv')
filtered_data['date'] = pd.to_datetime(filtered_data['date'])

# Дані
N = filtered_data['population'].iloc[0]  # загальна популяція
t_full = (filtered_data['date'] - filtered_data['date'].iloc[0]).dt.days.values  # дні з 2022-01-01

I_data = filtered_data['confirmed'].values  # інфіковані
R_data = filtered_data['recovered'].values  # одужалі

# Обчислення S_data (як залишок) і E_data (припустимо 0 на старті)
S_data = N - I_data - R_data  # сприйнятливі, припускаючи, що інші стани мінімальні
E_data = np.zeros(len(filtered_data))  # exposed (припустимо 0, оскільки даних немає)

# SEIRS модель
def seirs_model(y, t, beta, sigma, gamma, alpha):
    S, E, I, R = y
    dSdt = -beta * S * I / N + alpha * R  # Повернення R до S
    dEdt = beta * S * I / N - sigma * E  # Перехід E до I
    dIdt = sigma * E - gamma * I  # Перехід I до R
    dRdt = gamma * I - alpha * R  # Повернення R до S
    return dSdt, dEdt, dIdt, dRdt

# Функція втрат для фіксованого вікна
def loss_window(params, t_window, I_window, R_window, y0):
    beta, sigma, gamma, alpha = params
    sol = odeint(seirs_model, y0, t_window, args=(beta, sigma, gamma, alpha))
    I_pred, R_pred = sol[:, 2], sol[:, 3]
    w_I, w_R = 3.0, 1.0  # Вага для інфікованих вища
    return w_I * np.mean((I_window - I_pred)**2) + w_R * np.mean((R_window - R_pred)**2)

# Функція для запуску моделі з фіксованим вікном
def run_model(window_size=900, alpha_smoothing=0.7):
    params_history = []
    y0_history = [(S_data[0], E_data[0], I_data[0], R_data[0])]

    # Використовуємо одне фіксоване вікно
    i = 0
    t_window = t_full[i:i + window_size]
    I_window = I_data[i:i + window_size]
    R_window = R_data[i:i + window_size]
    
    if len(I_window) < window_size:
        t_window = t_window[:len(I_window)]
        I_window = I_window[:len(I_window)]
        R_window = R_window[:len(R_window)]
    
    y0 = y0_history[-1]
    
    bounds = [(0.05, 0.45), (0.1, 0.4), (0.03, 0.25), (0.001, 0.01)]
    initial_params = [0.3, 1/5.2, 0.1, 0.001]
    result = differential_evolution(lambda params: loss_window(params, t_window, I_window, R_window, y0), 
                                   bounds, maxiter=500, popsize=10)
    opt_beta, opt_sigma, opt_gamma, opt_alpha = result.x
    
    params_history.append((opt_beta, opt_sigma, opt_gamma, opt_alpha))
    sol = odeint(seirs_model, y0, t_window, args=(opt_beta, opt_sigma, opt_gamma, opt_alpha))
    y0_new = list(sol[-1, :])
    y0_history.append(tuple(y0_new))

    # Розширення прогнозу на весь період із фіксованими параметрами
    t_full_solution = t_full[:window_size]
    I_full_solution = I_data[:window_size].copy()
    R_full_solution = R_data[:window_size].copy()
    y0_current = y0_history[0]

    for i in range(window_size, len(t_full), window_size):
        t_window = t_full[i:i + window_size]
        if len(t_window) == 0:
            break
        sol = odeint(seirs_model, y0_current, t_window, args=(opt_beta, opt_sigma, opt_gamma, opt_alpha))
        t_full_solution = np.concatenate((t_full_solution, t_window))
        I_full_solution = np.concatenate((I_full_solution, sol[:, 2]))
        R_full_solution = np.concatenate((R_full_solution, sol[:, 3]))
        y0_current = list(sol[-1, :])

    return I_full_solution, R_full_solution, t_full_solution, params_history

# Запуск моделі з фіксованим вікном
I_final, R_final, t_final, params_history = run_model()

# Виведення MSE для усього періоду
mse = np.mean((I_data[:len(t_final)] - I_final)**2) + np.mean((R_data[:len(t_final)] - R_final)**2)
print(f"MSE для фіксованого вікна (window_size = 150): {mse}")

# Графіки основних результатів
plt.figure(figsize=(12, 6))
plt.plot(t_full, I_data, 'o', label='Реальні інфіковані')
plt.plot(t_final, I_final, '-', label='Модельні інфіковані')
plt.plot(t_full, R_data, 'o', label='Реальні одужалі')
plt.plot(t_final, R_final, '-', label='Модельні одужалі')
plt.xlabel('Дні з 2022-01-01')
plt.ylabel('Кількість людей')
plt.legend()
plt.title('SEIRS модель із фіксованим вікном (window_size = 150)')
plt.grid(True)
plt.show()

# Графік помилок
plt.figure(figsize=(12, 6))
plt.plot(t_final, I_data[:len(t_final)] - I_final, 'o-', label='Помилка для інфікованих')
plt.plot(t_final, R_data[:len(t_final)] - R_final, 'o-', label='Помилка для одужалих')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Дні')
plt.ylabel('Помилка (Дані - Прогноз)')
plt.legend()
plt.grid(True)
plt.show()

# Графіки параметрів (лише один набір, оскільки вікно фіксоване)
t_params = [0]  # Один набір параметрів
params = params_history[0]

plt.figure(figsize=(12, 6))
plt.plot(t_params, [params[0]], '-', label='β (коефіцієнт передачі)')
plt.xlabel('Дні з 2022-01-01')
plt.ylabel('Значення')
plt.title('Динаміка β')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t_params, [params[1]], '-', label='σ (швидкість переходу E→I)')
plt.xlabel('Дні з 2022-01-01')
plt.ylabel('Значення')
plt.title('Динаміка σ')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t_params, [params[2]], '-', label='γ (швидкість одужання)')
plt.xlabel('Дні з 2022-01-01')
plt.ylabel('Значення')
plt.title('Динаміка γ')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t_params, [params[3]], '-', label='α (швидкість втрати імунітету)')
plt.xlabel('Дні з 2022-01-01')
plt.ylabel('Значення')
plt.title('Динаміка α')
plt.grid(True)
plt.legend()
plt.show()