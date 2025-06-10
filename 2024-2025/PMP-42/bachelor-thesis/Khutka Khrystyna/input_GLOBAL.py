from scipy.integrate import odeint
import numpy as np

def model_func(y, t, beta, mu, gamma, alpha):
        S, E, I, R = y
        dSdt = -beta * S * I
        dEdt = beta * S * I - mu * E
        dIdt = -gamma * I + alpha * mu * E
        dRdt = gamma * I + (1 - alpha) * mu * E
        return [dSdt, dEdt, dIdt, dRdt]

def simulate_model(params):
    beta, mu, gamma, alpha = params
    N = 1
    E0, I0, R0 = 0.25, 0.05, 0
    S0 = N - I0 - R0 - E0
    y0 = [S0, E0, I0, R0]
    t = np.linspace(0, 500, 500)
    sol = odeint(model_func, y0, t, args=(beta, mu,gamma, alpha))
    return np.sum(sol[:, 2])

def model_to_plot(params):
    beta, mu, gamma, alpha = params
    N = 1
    E0, I0, R0 = 0.25, 0.05, 0
    S0 = N - I0 - R0 - E0
    y0 = [S0, E0, I0, R0]
    t = np.linspace(0, 1000, 1000)
    sol = odeint(model_func, y0, t, args=(beta, mu,gamma, alpha))
    return sol[:, 2]
# Bounds
param_bounds = [
    [0.1, 0.3], 
    [0.005, 0.01], 
    [0.005, 0.01], 
    [0.2, 0.5]
]
t_span = (0, 500, 500)
n_train = 65







#-----------------------BRANIN FUNCTION-----------------------
# def simulate_model(params):
#     b = 5.1 / (4 * np.pi**2)
#     c = 5 / np.pi
#     r = 6.0
#     s = 10.0
#     t = 1 / (8 * np.pi)
#     x1 = params[0]
#     x2 = params[1]
#     return (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

# param_bounds = [[-5, 10], [0, 15]]

# def model_to_plot(params):
#     b = 5.1 / (4 * np.pi**2)
#     c = 5 / np.pi
#     r = 6.0
#     s = 10.0
#     t = 1 / (8 * np.pi)
#     x1 = params[0]
#     x2 = params[1]
#     return (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s








#-------------------SEIR ADAPTIVE----------------------
# import numpy as np
# from scipy.integrate import odeint


# def beta(a):
#     """
#     SMOOTH exposure rate (transmission).
#     Uses a sigmoid function to model how the transmission rate grows with virality.
#     It ranges from a low base rate to a high rate for very viral news.
#     """
#     base_rate = 0.1
#     max_rate = 0.9
#     return base_rate + (max_rate - base_rate) * (1 / (1 + np.exp(-10 * (a - 0.5))))

# def alpha(a):
#     """
#     SMOOTH fraction of exposed who become spreaders.
#     More viral news (higher 'a') makes people more likely to share.
#     """
#     min_share_fraction = 0.1
#     max_share_fraction = 0.8
#     return min_share_fraction + (max_share_fraction - min_share_fraction) * a**2

# zeta = 0.2
# gamma = 0.1


# def sir_model(y, t, a):
#     """
#     SEIR-like ODE system for fake news spread with SMOOTH parameters.
#     y = [S, E, I, R],  a = 'virality_index' in [0,1]
#     """
#     S, E, I, R = y
#     b = beta(a)
#     al = alpha(a)
#     dSdt = -b * S * I
#     dEdt = b * S * I - zeta * E
#     dIdt = zeta * al * E - gamma * I
#     dRdt = zeta * (1 - al) * E + gamma * I
#     return [dSdt, dEdt, dIdt, dRdt]


# def simulate_model(a, t_max=500, N=1.0, E0=0.05, I0=0.01, R0=0.0):
#     """
#     Simulates the SEIR system over [0, t_max] for a given virality_index 'a'.
#     Returns the cumulative infected measure: int_0^T I(t) dt.
#     """
#     a = float(a)
#     S0 = N - E0 - I0 - R0
#     y0 = [S0, E0, I0, R0]
#     t = np.linspace(0, t_max, t_max + 1)
    
#     sol = odeint(sir_model, y0, t, args=(a,))
#     return np.trapz(sol[:, 2], t)

# param_bounds = np.array([[0.0, 1.0]])