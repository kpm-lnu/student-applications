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
S_data = N - I_data - R_data  # Сприйнятливі
t_full = (filtered_data['date'] - filtered_data['date'].iloc[0]).dt.days.values

# SEIR модель
def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Функція втрат для вікна
def loss_window(params, t_window, I_window, R_window, y0):
    beta, sigma, gamma = params
    sol = odeint(seir_model, y0, t_window, args=(beta, sigma, gamma))
    I_pred, R_pred = sol[:, 2], sol[:, 3]
    w_I, w_R = 3.0, 1.0
    return w_I * np.mean((I_window - I_pred)**2) + w_R * np.mean((R_window - R_pred)**2)

# Функція для запуску моделі з рухомими вікнами
def run_model(window_size, step_size=60, alpha=0.7):
    params_history = []
    y0_history = [(S_data[0], 1000, I_data[0], R_data[0])]

    for i in range(0, len(t_full) - window_size + 1, step_size):
        t_window = t_full[i:i + window_size]
        I_window = I_data[i:i + window_size]
        R_window = R_data[i:i + window_size]
        
        if len(I_window) < window_size:
            t_window = t_window[:len(I_window)]
            I_window = I_window[:len(I_window)]
            R_window = R_window[:len(R_window)]
        
        y0 = y0_history[-1]
		
        bounds = [(0.05, 0.45), (0.1, 0.4), (0.03, 0.25)]
        initial_params = [0.3, 1/5.2, 1/10] if not params_history else params_history[-1]
        result = differential_evolution(lambda params: loss_window(params, t_window, I_window, R_window, y0), 
                                       bounds, maxiter=500, popsize=10)
        opt_beta, opt_sigma, opt_gamma = result.x
        
        if params_history:
            opt_beta = alpha * opt_beta + (1 - alpha) * params_history[-1][0]
            opt_sigma = alpha * opt_sigma + (1 - alpha) * params_history[-1][1]
            opt_gamma = alpha * opt_gamma + (1 - alpha) * params_history[-1][2]
        
        params_history.append((opt_beta, opt_sigma, opt_gamma))
        sol = odeint(seir_model, y0, t_window, args=(opt_beta, opt_sigma, opt_gamma))
        y0_new = list(sol[-1, :])
        y0_history.append(tuple(y0_new))

    t_full_solution = t_full[:window_size]
    I_full_solution = I_data[:window_size].copy()
    R_full_solution = R_data[:window_size].copy()
    y0_current = y0_history[0]

    for i, params in enumerate(params_history):
        opt_beta, opt_sigma, opt_gamma = params
        start_idx = len(t_full_solution)
        end_idx = min(start_idx + step_size, len(t_full))
        t_window = t_full[start_idx:end_idx]
        if len(t_window) == 0:
            break
        sol = odeint(seir_model, y0_current, t_window, args=(opt_beta, opt_sigma, opt_gamma))
        t_full_solution = np.concatenate((t_full_solution, t_window))
        I_full_solution = np.concatenate((I_full_solution, sol[:, 2]))
        R_full_solution = np.concatenate((R_full_solution, sol[:, 3]))
        y0_current = list(sol[-1, :])

    return I_full_solution, R_full_solution, t_full_solution, params_history

# Дослідження оптимального window_size і step_size
window_sizes = range(20, 151, 5)  # Тестуємо від 20 до 150 з кроком 10
step_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]  # Тестуємо різні значення step_size
best_mse = float('inf')
best_params = None

# Розбиття на тренувальну та валідаційну вибірки
train_end = 350
t_train = t_full[t_full <= train_end]
I_train = I_data[t_full <= train_end]
R_train = R_data[t_full <= train_end]
t_val = t_full[t_full > train_end]
I_val = I_data[t_full > train_end]
R_val = R_data[t_full > train_end]

# Тестування комбінацій window_size і step_size з умовою |window_size - step_size| <= 10
for window_size in window_sizes:
    for step_size in step_sizes:
        if abs(window_size - step_size) <= 10:  # Умова для близькості
            I_pred, R_pred, t_pred, _ = run_model(window_size, step_size)
            
            # Фільтрація прогнозу для валідаційного періоду
            mask_val = t_pred > train_end
            t_val_pred = t_pred[mask_val]
            I_pred_val = I_pred[mask_val]
            R_pred_val = R_pred[mask_val]
            
            # Перевірка відповідності розмірів
            if len(t_val_pred) > 0 and len(I_val) == len(I_pred_val):
                mse = np.mean((I_val - I_pred_val)**2) + np.mean((R_val - R_pred_val)**2)
                print(f"window_size = {window_size}, step_size = {step_size}, MSE = {mse}")
                if mse < best_mse:
                    best_mse = mse
                    best_params = {'window_size': window_size, 'step_size': step_size}
            else:
                print(f"window_size = {window_size}, step_size = {step_size}: Невідповідність розмірів, MSE не обчислено")

if best_params is None:
    print("Не вдалося знайти оптимальні параметри через невідповідність розмірів.")
else:
    print(f"Оптимальні параметри: window_size = {best_params['window_size']}, step_size = {best_params['step_size']} з MSE: {best_mse}")

    # Запуск із найкращими параметрами для повного графіка
    I_final, R_final, t_final, params_history = run_model(best_params['window_size'], best_params['step_size'])

    # Графіки основних результатів
    plt.figure(figsize=(12, 6))
    plt.plot(t_full, I_data, 'o', label='Реальні інфіковані (active)')
    plt.plot(t_final, I_final, '-', label='Модельні інфіковані')
    plt.plot(t_full, R_data, 'o', label='Реальні одужалі + померлі')
    plt.plot(t_final, R_final, '-', label='Модельні одужалі')
    plt.xlabel('Дні з 21.03.2020')
    plt.ylabel('Кількість людей')
    plt.legend()
    plt.title(f'SEIR модель із рухомими вікнами (window_size = {best_params["window_size"]}, step_size = {best_params["step_size"]})')
    plt.grid(True)
    plt.show()

    # Графік помилок
    plt.figure(figsize=(12, 6))
    plt.plot(t_final, I_data[:len(t_final)] - I_final, 'o-', label='Помилка для інфікованих')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Дні')
    plt.ylabel('Помилка (I_data - I_pred)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Графіки параметрів
    t_params = np.arange(0, len(params_history) * best_params['step_size'], best_params['step_size'])[:len(params_history)]

    plt.figure(figsize=(12, 6))
    plt.plot(t_params, [p[0] for p in params_history], '-', label='β (коефіцієнт передачі)')
    plt.xlabel('Дні з 21.03.2020')
    plt.ylabel('Значення')
    plt.title('Динаміка β')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(t_params, [p[1] for p in params_history], '-', label='σ (швидкість переходу E→I)')
    plt.xlabel('Дні з 21.03.2020')
    plt.ylabel('Значення')
    plt.title('Динаміка σ')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(t_params, [p[2] for p in params_history], '-', label='γ (швидкість одужання)')
    plt.xlabel('Дні з 21.03.2020')
    plt.ylabel('Значення')
    plt.title('Динаміка γ')
    plt.grid(True)
    plt.legend()
    plt.show()