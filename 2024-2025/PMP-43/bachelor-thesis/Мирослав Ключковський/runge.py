import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from tabulate import tabulate

# Параметри моделі
N = 41e6  # Загальне населення
beta = 0.3  # рівень зараження
sigma = 0.2  # інверсія інкубаційного періоду
gamma = 0.1  # інверсія періоду одужання
nu = 0.0  # швидкість вакцинації
Ds, De, Di, Dr = 0.1, 0.1, 0.1, 0.1  # Коефіцієнти дифузії

# Початкові умови (абсолютні значення)
I0 = 18  # Початкові інфіковані
E0 = 0
R0 = 0
S0 = N - I0 - E0 - R0
Nx, Ny = 50, 50  # Розмірність простору
T = 730  # Часова межа (2 роки)
dt = 1  # Крок інтегрування
steps = int(T / dt)

# Ініціалізація полів (нормовані на загальну популяцію)
S = np.ones((Nx, Ny)) * S0 / N
E = np.ones((Nx, Ny)) * E0 / N
I = np.ones((Nx, Ny)) * I0 / N
R = np.ones((Nx, Ny)) * R0 / N
E[Nx // 2, Ny // 2] = 0.0  # Початковий спалах у центрі

# Масиви для збереження абсолютних значень
S_abs, E_abs, I_abs, R_abs = [], [], [], []


def runge_kutta_4(S, E, I, R, dt):
    # Нормалізація перед обчисленням
    S_norm = S * N
    I_norm = I * N

    dSdt = (-beta * S_norm * I_norm / N + nu * R * N + Ds * laplace(S, mode='wrap')) / N
    dEdt = (beta * S_norm * I_norm / N - sigma * E * N + De * laplace(E, mode='wrap')) / N
    dIdt = (sigma * E * N - gamma * I * N + Di * laplace(I, mode='wrap')) / N
    dRdt = (gamma * I * N - nu * R * N + Dr * laplace(R, mode='wrap')) / N

    # Коефіцієнти Рунге-Кутта (аналогічно до попереднього коду)
    k1_S, k1_E, k1_I, k1_R = dSdt * dt, dEdt * dt, dIdt * dt, dRdt * dt
    k2_S = (dSdt + 0.5 * k1_S) * dt
    k2_E = (dEdt + 0.5 * k1_E) * dt
    k2_I = (dIdt + 0.5 * k1_I) * dt
    k2_R = (dRdt + 0.5 * k1_R) * dt

    k3_S = (dSdt + 0.5 * k2_S) * dt
    k3_E = (dEdt + 0.5 * k2_E) * dt
    k3_I = (dIdt + 0.5 * k2_I) * dt
    k3_R = (dRdt + 0.5 * k2_R) * dt

    k4_S = (dSdt + k3_S) * dt
    k4_E = (dEdt + k3_E) * dt
    k4_I = (dIdt + k3_I) * dt
    k4_R = (dRdt + k3_R) * dt

    S_new = S + (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6
    E_new = E + (k1_E + 2 * k2_E + 2 * k3_E + k4_E) / 6
    I_new = I + (k1_I + 2 * k2_I + 2 * k3_I + k4_I) / 6
    R_new = R + (k1_R + 2 * k2_R + 2 * k3_R + k4_R) / 6

    return S_new, E_new, I_new, R_new


# Основний цикл
S_abs = [np.mean(S) * N]
E_abs = [np.mean(E) * N]
I_abs = [np.mean(I) * N]
R_abs = [np.mean(R) * N]

for _ in range(1,steps):
    S, E, I, R = runge_kutta_4(S, E, I, R, dt)

    # Збереження абсолютних значень
    S_abs.append(np.mean(S) * N)
    E_abs.append(np.mean(E) * N)
    I_abs.append(np.mean(I) * N)
    R_abs.append(np.mean(R) * N)

# Візуалізація
plt.figure(figsize=(10, 5))
time = np.linspace(0, T, steps)
plt.plot(time, S_abs, label='S (сприйнятливі)')
plt.plot(time, E_abs, label='E (експоновані)')
plt.plot(time, I_abs, label='I (інфіковані)')
plt.plot(time, R_abs, label='R (одужалі)')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість людей')
plt.legend()
plt.grid()
plt.title('Динаміка SEIR-моделі у часі')
plt.show()

# Виведення таблиці
headers = ["Time (days)", "Susceptible", "Exposed", "Infectious", "Recovered"]
table_data = list(zip(
    time.astype(int),
    np.round(S_abs).astype(int),
    np.round(E_abs).astype(int),
    np.round(I_abs).astype(int),
    np.round(R_abs).astype(int)
))

print(tabulate(table_data[:100], headers=headers, tablefmt="grid", floatfmt=",.0f"))
