import numpy as np
import matplotlib.pyplot as plt
from seir_model import SEIR_model, calc_SEIR
from seir_fit import parameter_estimation
from scipy.integrate import odeint
from generator import generate


def plot_seir_model(params):
    S0 = 1
    E0 = 0
    I0 = 0.01
    R0 = 0
    y0 = [S0, E0, I0, R0]
    beta, gamma, alpha, delta, eta = params

    
    t = np.linspace(0, 200, 201)
    solution = odeint(SEIR_model, y0, t, args=(beta, gamma, alpha, delta, eta))
    S, E, I, R = solution.T * 1000
    merged = E + I

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()



data = generate()

calc_SEIR()

#estimated_params = parameter_estimation(data)

#plot_seir_model(estimated_params)