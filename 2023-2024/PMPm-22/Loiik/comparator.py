import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from sir_model import SIR_model
from seir_model import SEIR_model


def norm(a, b):
    a = np.array(a)
    b = np.array(b)

    #squared_diff = np.sqrt(np.sum((a - b) ** 2))
    squared_diff = 0
    for i in range(3):
        for j in range(len(b[0])):
            dx = a[i][j] - b[i][j]
            squared_diff += dx * dx
    return np.sqrt(squared_diff)


def loss_function(params, generated_data, t):
    y0 = [1, 0.01, 0]  # Initial conditions
    beta, alpha = params
    solution = odeint(SIR_model, y0, t, args=(beta, alpha))
    solution = solution.T
    diffs = generated_data - solution
    loss = np.sum(diffs ** 2)
    return loss


def fit_parameters(generated_data, t, initial_guess):
    result = minimize(loss_function, initial_guess, args=(generated_data, t), method='Nelder-Mead')
    best_params = result.x
    return best_params, result.fun


def best_parameter_estimation(generated_data, t):
    best_loss = 10**47#float('inf')
    best_params = None
    
    xyz = [0, 0]
    for beta_guess in np.arange(0, 1, 0.1):
        for alpha_guess in np.arange(0.01, 1, 0.1):
            initial_guess = [beta_guess, alpha_guess]
            estimated_params, loss = fit_parameters(generated_data, t, initial_guess)
            if loss < best_loss:
                best_loss = loss
                best_params = estimated_params
                xyz = [beta_guess, alpha_guess]
    print('Best initial guess:')
    print(xyz)
    return best_params


def parameter_estimation(generated_data):
    t = np.linspace(0, 200, 201)
    estimated_params = best_parameter_estimation(generated_data, t)
    print("Estimated parameters for SIR:", estimated_params)
    return estimated_params