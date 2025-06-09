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

def euler_explicit(y, t, dt, N, beta, sigma, gamma):
    dydt = np.array(derivatives(y, t, N, beta, sigma, gamma))
    y_next = y + dt * dydt
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

# Інтегрування методом Ейлера
for i in range(1, len(t)):
    results[i] = euler_explicit(results[i-1], t[i-1], dt, N, beta, sigma, gamma)

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
plt.title('SEIR Model Dynamics (Euler Method)')
plt.legend()
plt.show()
