import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
beta = 0.4       # transmission rate
sigma = 0.1786      # rate of progression from exposed to infected
gamma = 0.0786      # recovery rate
eta = 0.005       # vaccination rate
epsilon_R = 0.0083 # rate of loss of immunity for recovered individuals
epsilon_V = 0.0056 # rate of loss of immunity for vaccinated individuals

S0 = 0.9   # initial proportion of susceptible individuals
E0 = 0.05  # initial proportion of exposed individuals
I0 = 0.05  # initial proportion of infected individuals
R0 = 0.0     # initial proportion of recovered individuals
V0 = 0.0     # initial proportion of vaccinated individuals


# Time grid
t = np.linspace(0, 365, 365)  # 160 days

# SEIRV model differential equations
def deriv(y, t, beta, sigma, gamma, eta, epsilon_R, epsilon_V):
    S, E, I, R, V = y
    dSdt = -beta * S * I - eta * S + epsilon_R * R
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I + epsilon_V * V - epsilon_R * R
    dVdt = eta * (S) - epsilon_V * V
    return dSdt, dEdt, dIdt, dRdt, dVdt

# Initial conditions vector
y0 = S0, E0, I0, R0, V0

# Integrate the SEIRV equations over the time grid, t
ret = odeint(deriv, y0, t, args=(beta, sigma, gamma, eta, epsilon_R, epsilon_V))
S, E, I, R, V = ret.T
#print(I[162])
# Plot the data
fig = plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
plt.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
plt.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
plt.plot(t, V, 'c', alpha=0.7, linewidth=2, label='Vaccinated')
plt.xlabel('Time (days)')
plt.ylabel('Proportion')
plt.title('SEIRV Model with Waning Immunity')
plt.legend(loc='best')
plt.grid(True)
plt.show()
