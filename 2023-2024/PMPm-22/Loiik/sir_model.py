import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy.linalg import eig


def SIR_model(y, t, beta, alpha):
    S, I, R = y
    dSdt = -beta*S*I
    dIdt = beta*S*I-alpha*I
    dRdt = alpha * I
    return [dSdt, dIdt, dRdt]


def calc_SIR(params):
    S0 = 1
    I0 = 0.03
    R0 = 0
    y0 = [S0, I0, R0]
    
    beta, alpha = params
    t = np.linspace(0, 200, 201)
    solution = odeint(SIR_model, y0, t, args=(beta, alpha))
    S, I, R = solution.T

    plt.figure(figsize=(10, 6))
    plt.plot(t, S * 1000, label='Susceptible')
    plt.plot(t, I * 1000, label='Infected')
    plt.plot(t, R * 1000, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title("SIR")
    plt.legend()
    plt.grid(True)
    plt.show()

    return np.array((S, I, R))


