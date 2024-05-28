import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SEIR_model(y, t, beta, gamma, alpha, delta, eta):
    S, E, I, R = y
    dSdt = -beta * S * I + (1 - alpha) * gamma * I
    dEdt = beta * S * I - delta * E
    dIdt = delta * (1 - eta) * E - gamma * I
    dRdt = delta * eta * E + alpha * gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def calc_SEIR():
    S0 = 1
    E0 = 0
    I0 = 0.03
    R0 = 0
    y0 = [S0, E0, I0, R0]
    
    beta = 0.2937
    gamma = 0.0079
    alpha = 0.9993
    delta = 0.0336
    eta = 0.01
    
    t = np.linspace(0, 200, 201)
    solution = odeint(SEIR_model, y0, t, args=(beta, gamma, alpha, delta, eta))
    S, E, I, R = solution.T * 1000
    merged = E + I
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infectious')
    # plt.plot(t, merged, label='Exposed and Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()
    return np.array((S, merged, R))

