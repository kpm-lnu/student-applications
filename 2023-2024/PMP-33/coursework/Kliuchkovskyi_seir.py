import numpy as npimport matplotlib.pyplot as plt
from tabulate import tabulate
def derivatives(y, t, N, beta, sigma, gamma):    S, E, I, R = y
    dSdt = -beta * S * I / N    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt
def runge_kutta_4(y, t, dt, N, beta, sigma, gamma):    k1 = dt * np.array(derivatives(y, t, N, beta, sigma, gamma))
    k2 = dt * np.array(derivatives(y + 0.5 * k1, t + 0.5 * dt, N, beta, sigma, gamma))    k3 = dt * np.array(derivatives(y + 0.5 * k2, t + 0.5 * dt, N, beta, sigma, gamma))
    k4 = dt * np.array(derivatives(y + k3, t + dt, N, beta, sigma, gamma))    y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y_next
N = 100I0 = 1
E0 = 0R0 = 0
S0 = N - I0 - R0 - E0dt = 1
t = np.arange(0, 55, dt)
beta = 1sigma = 1
gamma = 0.1
results = np.zeros((len(t), 4))y0 = S0, E0, I0, R0
results[0] = y0
for i in range(1, len(t)):    results[i] = runge_kutta_4(results[i-1], t[i-1], dt, N, beta, sigma, gamma)
# Construct table headers
headers = ["Time", "Susceptible", "Exposed", "Infectious", "Recovered"]# Combine time and results into one list
table_data = [[t[i], results[i][0], results[i][1], results[i][2], results[i][3]] for i in range(len(t))]# Print the table
print(tabulate(table_data, headers=headers, tablefmt="grid"))
# Plotting the resultsplt.plot(t, results[:, 0], label='Susceptible')
plt.plot(t, results[:, 1], label='Exposed')plt.plot(t, results[:, 2], label='Infectious')
plt.plot(t, results[:, 3], label='Recovered')plt.xlabel('Time')
plt.ylabel('Number of Individuals')plt.title('SEIR Model Simulation')
plt.legend()plt.show()