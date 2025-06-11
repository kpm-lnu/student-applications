import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

filtered_data = pd.read_csv('input/SEIRVD.csv')
filtered_data['date'] = pd.to_datetime(filtered_data['date'])

S_data = filtered_data['susceptible'].values
E_data = filtered_data['exposed'].values
I_data = filtered_data['infected'].values
R_data = filtered_data['recovered'].values
V_data = filtered_data['people_vaccinated'].values
D_data = filtered_data['deaths_x'].values
N = 41_000_000
t_full = (filtered_data['date'] - filtered_data['date'].iloc[0]).dt.days.values

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

# Функція втрат для вікна
def loss_window(params, t_window, I_window, R_window, V_window, D_window, y0):
    beta, mu, v_rate, E0 = params
    sigma = 1 / 5.2  # Фіксований період інкубації
    gamma = 1 / 10   # Фіксований період одужання
    y0_adjusted = [y0[0], E0, y0[2], y0[3], y0[4], y0[5]]
    sol = odeint(seirvd_model_conditional_vacc, y0_adjusted, t_window, args=(beta, sigma, gamma, mu, v_rate))
    I_pred = sol[:, 2]
    R_pred = sol[:, 3]
    V_pred = sol[:, 4]
    D_pred = sol[:, 5]
    return (3.0 * np.mean((I_window - I_pred) ** 2) + 
            1.5 * np.mean((R_window - R_pred) ** 2) + 
            3.0 * np.mean((V_window - V_pred) ** 2) + 
            1.0 * np.mean((D_window - D_pred) ** 2))

# Функція для запуску моделі з рухомими вікнами
def run_model(window_size=150, step_size=50, alpha=0.8):
    params_history = []
    y0_history = [(S_data[0], E_data[0], I_data[0], R_data[0], V_data[0], D_data[0])]

    for i in range(0, len(t_full) - window_size + 1, step_size):
        t_window = t_full[i:i + window_size]
        I_window = I_data[i:i + window_size]
        R_window = R_data[i:i + window_size]
        V_window = V_data[i:i + window_size]
        D_window = D_data[i:i + window_size]
        
        if len(I_window) < window_size:
            t_window = t_window[:len(I_window)]
            I_window = I_window[:len(I_window)]
            R_window = R_window[:len(R_window)]
            V_window = V_window[:len(V_window)]
            D_window = D_window[:len(D_window)]
        
        y0 = y0_history[-1]
        
        bounds = [(0.0001, 1), (0.00001, 0.05), (0.00001, 0.01), (0, 10)]
        initial_params = [0.3, 0.005, 0.0005, 2000] if not params_history else params_history[-1]
        result = minimize(lambda params: loss_window(params, t_window, I_window, R_window, V_window, D_window, y0), 
                          initial_params, bounds=bounds)
        opt_beta, opt_mu, opt_vrate, opt_E0 = result.x
        
        if params_history:
            opt_beta = alpha * opt_beta + (1 - alpha) * params_history[-1][0]
            opt_mu = alpha * opt_mu + (1 - alpha) * params_history[-1][1]
            opt_vrate = alpha * opt_vrate + (1 - alpha) * params_history[-1][2]
            opt_E0 = alpha * opt_E0 + (1 - alpha) * params_history[-1][3]
        
        params_history.append((opt_beta, opt_mu, opt_vrate, opt_E0))
        sigma = 1 / 5.2
        gamma = 1 / 10
        sol = odeint(seirvd_model_conditional_vacc, y0, t_window, args=(opt_beta, sigma, gamma, opt_mu, opt_vrate))
        y0_new = list(sol[-1, :])
        y0_history.append(tuple(y0_new))

    t_full_solution = t_full[:window_size]
    I_full_solution = I_data[:window_size].copy()
    R_full_solution = R_data[:window_size].copy()
    V_full_solution = V_data[:window_size].copy()
    D_full_solution = D_data[:window_size].copy()
    y0_current = y0_history[0]

    for i, params in enumerate(params_history):
        opt_beta, opt_mu, opt_vrate, opt_E0 = params
        start_idx = len(t_full_solution)
        end_idx = min(start_idx + step_size, len(t_full))
        t_window = t_full[start_idx:end_idx]
        if len(t_window) == 0:
            break
        sigma = 1 / 5.2
        gamma = 1 / 10
        sol = odeint(seirvd_model_conditional_vacc, y0_current, t_window, args=(opt_beta, sigma, gamma, opt_mu, opt_vrate))
        t_full_solution = np.concatenate((t_full_solution, t_window))
        I_full_solution = np.concatenate((I_full_solution, sol[:, 2]))
        R_full_solution = np.concatenate((R_full_solution, sol[:, 3]))
        V_full_solution = np.concatenate((V_full_solution, sol[:, 4]))
        D_full_solution = np.concatenate((D_full_solution, sol[:, 5]))
        y0_current = list(sol[-1, :])

    return I_full_solution, R_full_solution, V_full_solution, D_full_solution, t_full_solution, params_history

# Запуск моделі
I_final, R_final, V_final, D_final, t_final, params_history = run_model(window_size=150, step_size=50, alpha=0.8)

# Виведення MSE
mse = (3.0 * np.mean((I_data[:len(t_final)] - I_final) ** 2) + 
       1.5 * np.mean((R_data[:len(t_final)] - R_final) ** 2) + 
       1.0 * np.mean((V_data[:len(t_final)] - V_final) ** 2) + 
       1.0 * np.mean((D_data[:len(t_final)] - D_final) ** 2))
print(f"MSE для моделі: {mse}")

# Графіки основних результатів
plt.figure(figsize=(14, 8))
plt.plot(t_full, I_data, 'o', label='Реальні інфіковані')
plt.plot(t_final, I_final, '-', label='Модель: інфіковані')
plt.plot(t_full, R_data, 'o', label='Реальні одужалі')
plt.plot(t_final, R_final, '-', label='Модель: одужалі')
plt.plot(t_full, V_data, 'o', label='Реальні вакциновані')
plt.plot(t_final, V_final, '-', label='Модель: вакциновані')
plt.plot(t_full, D_data, 'o', label='Реальні померлі')
plt.plot(t_final, D_final, '-', label='Модель: померлі')
plt.xlabel('Дні з 21.03.2020')
plt.ylabel('Кількість людей')
plt.title('SEIRVD модель з рухомими вікнами (window_size = 150, step_size = 50, alpha = 0.8)')
plt.legend()
plt.grid(True)
plt.show()

# Графік помилок
plt.figure(figsize=(12, 6))
plt.plot(t_final, I_data[:len(t_final)] - I_final, 'o-', label='Помилка для інфікованих')
plt.plot(t_final, R_data[:len(t_final)] - R_final, 'o-', label='Помилка для одужалих')
plt.plot(t_final, V_data[:len(t_final)] - V_final, 'o-', label='Помилка для вакцинованих')
plt.plot(t_final, D_data[:len(t_final)] - D_final, 'o-', label='Помилка для померлих')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Дні')
plt.ylabel('Помилка (Дані - Прогноз)')
plt.legend()
plt.grid(True)
plt.show()

# Графіки параметрів
t_params = np.arange(0, len(params_history) * 50, 50)[:len(params_history)]

plt.figure(figsize=(12, 6))
plt.plot(t_params, [p[0] for p in params_history], '-', label='β (коефіцієнт передачі)')
plt.xlabel('Дні з 21.03.2020')
plt.ylabel('Значення')
plt.title('Динаміка β')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t_params, [p[1] for p in params_history], '-', label='μ (швидкість смертності)')
plt.xlabel('Дні з 21.03.2020')
plt.ylabel('Значення')
plt.title('Динаміка μ')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t_params, [p[2] for p in params_history], '-', label='v_rate (швидкість вакцинації)')
plt.xlabel('Дні з 21.03.2020')
plt.ylabel('Значення')
plt.title('Динаміка v_rate')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t_params, [p[3] for p in params_history], '-', label='E0 (початкова кількість схильних)')
plt.xlabel('Дні з 21.03.2020')
plt.ylabel('Значення')
plt.title('Динаміка E0')
plt.grid(True)
plt.legend()
plt.show()