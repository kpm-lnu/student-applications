import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from seir_model import SEIR_model, calc_SEIR


def loss_function(params, generated_data, t):
    y0 = [1, 0, 0.01, 0]
    beta, gamma, alpha, delta, eta = params
    solution = odeint(SEIR_model, y0, t, args=(beta, gamma, alpha, delta, eta))
    solution = solution.T
    s = solution[0]
    i = solution[1]+solution[2]
    r = solution[3]
    diffs = generated_data - np.array((s, i, r))
    loss = np.sum(abs(diffs))
    return loss


def fit_parameters(generated_data, t, initial_guess):
    result = minimize(loss_function, initial_guess, args=(generated_data, t), method='Nelder-Mead')
    best_params = result.x
    return best_params, result.fun


def best_parameter_estimation(generated_data, t):
    best_loss = float('inf')
    best_params = None

    xyz = [0, 0, 0, 0, 0]
    for beta_guess in np.arange(0, 0.35, 0.1):
        for gamma_guess in np.arange(0, 0.35, 0.1):
            for alpha_guess in np.arange(0, 0.35, 0.1):
                for delta_guess in np.arange(0, 0.35, 0.1):
                    for eta_guess in np.arange(0, 0.35, 0.1):
                        initial_guess = [beta_guess, gamma_guess, alpha_guess, delta_guess, eta_guess]
                        estimated_params, loss = fit_parameters(generated_data, t, initial_guess)
                        if loss < best_loss:
                            best_loss = loss
                            best_params = estimated_params
                            xyz = [beta_guess, gamma_guess, alpha_guess, delta_guess, eta_guess]
    print('Best initial guess:')
    print(xyz)
    return best_params


def parameter_estimation(generated_data):
    t = np.linspace(0, 200, 201)
    estimated_params = best_parameter_estimation(generated_data, t)
    SEIR_model_res = calc_SEIR()
    print("Estimated parameters for SEIR:", estimated_params)
    return estimated_params