import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def derivatives(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def runge_kutta_4(y, t, dt, N, beta, sigma, gamma):
    k1 = dt * np.array(derivatives(y, t, N, beta, sigma, gamma))
    k2 = dt * np.array(derivatives(y + 0.5 * k1, t + 0.5 * dt, N, beta, sigma, gamma))
    k3 = dt * np.array(derivatives(y + 0.5 * k2, t + 0.5 * dt, N, beta, sigma, gamma))
    k4 = dt * np.array(derivatives(y + k3, t + dt, N, beta, sigma, gamma))
    y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y_next

# Параметри моделі
N = 41e6
I0 = 18
E0 = 0
R0 = 0
S0 = N - I0 - R0 - E0
dt = 1
t = np.arange(0, 730, dt)

beta = 0.3
sigma = 0.2
gamma = 0.1

# Ініціалізація результатів
results = np.zeros((len(t), 4))
y0 = S0, E0, I0, R0
results[0] = y0

# Інтегрування
for i in range(1, len(t)):
    results[i] = runge_kutta_4(results[i-1], t[i-1], dt, N, beta, sigma, gamma)

# Форматування таблиці
headers = ["Time (days)", "Susceptible", "Exposed", "Infectious", "Recovered"]
table_data = [[int(t[i]),
               int(round(results[i][0])),
               int(round(results[i][1])),
               int(round(results[i][2])),
               int(round(results[i][3]))]
              for i in range(len(t))]

# Виведення перших 10 днів
print(tabulate(table_data[:100], headers=headers, tablefmt="grid"))

# Візуалізація
plt.plot(t, results[:, 0], label='Susceptible')
plt.plot(t, results[:, 1], label='Exposed')
plt.plot(t, results[:, 2], label='Infectious')
plt.plot(t, results[:, 3], label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of People')
plt.title('SEIR Model Dynamics')
plt.legend()
plt.show()