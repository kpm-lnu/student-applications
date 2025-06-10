import numpy as np
from scipy.integrate import odeint


def beta(a):
    """
    SMOOTH exposure rate (transmission).
    Uses a sigmoid function to model how the transmission rate grows with virality.
    It ranges from a low base rate to a high rate for very viral news.
    """
    base_rate = 0.1
    max_rate = 0.9
    return base_rate + (max_rate - base_rate) * (1 / (1 + np.exp(-10 * (a - 0.5))))

def alpha(a):
    """
    SMOOTH fraction of exposed who become spreaders.
    More viral news (higher 'a') makes people more likely to share.
    """
    min_share_fraction = 0.1
    max_share_fraction = 0.8
    return min_share_fraction + (max_share_fraction - min_share_fraction) * a**2

zeta = 0.2
gamma = 0.1


def sir_model(y, t, a):
    """
    SEIR-like ODE system for fake news spread with SMOOTH parameters.
    y = [S, E, I, R],  a = 'virality_index' in [0,1]
    """
    S, E, I, R = y
    
    b = beta(a)
    al = alpha(a)

    dSdt = -b * S * I
    dEdt = b * S * I - zeta * E
    dIdt = zeta * al * E - gamma * I
    dRdt = zeta * (1 - al) * E + gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def simulate_model(a, t_max=200, N=1.0, E0=0.05, I0=0.01, R0=0.0):
    """
    Simulates the SEIR system over [0, t_max] for a given virality_index 'a'.
    Returns the cumulative infected measure: int_0^T I(t) dt.
    """
    a = float(a)

    S0 = N - E0 - I0 - R0
    y0 = [S0, E0, I0, R0]
    t = np.linspace(0, t_max, t_max + 1)
    
    sol = odeint(sir_model, y0, t, args=(a,))
    
    return np.trapz(sol[:, 2], t)


param_bounds = np.array([[0.0, 1.0]])