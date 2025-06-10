import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def seirv_model(t, y, beta, beta_v, sigma, gamma, nu):
    S, E, I, R, V = y
    
    dSdt = -beta * S * I - nu * S
    dEdt = beta * S * I + beta_v * V * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dVdt = nu * S - beta_v * V * I
    
    return [dSdt, dEdt, dIdt, dRdt, dVdt]

def run_seirv_model(beta, beta_v, sigma, gamma, nu, I0, V0, t_max=200, dt=0.1):
    S0 = 1 - I0 - V0
    E0 = 0
    R0 = 0
    y0 = [S0, E0, I0, R0, V0]
    
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    sol = solve_ivp(
        seirv_model, t_span, y0, args=(beta, beta_v, sigma, gamma, nu),
        method='RK45', t_eval=t_eval
    )
    
    return sol.t, sol.y

def compute_outputs(t, y):
    S, E, I, R, V = y
    
    I_max = np.max(I)
    t_max = t[np.argmax(I)]
    
    R_final = R[-1]
    
    V0 = V[0]
    V_final = V[-1]
    V_infected = V0 - V_final
    
    return {
        'I_max': I_max,
        't_max': t_max,
        'R_final': R_final,
        'V_infected': V_infected
    }

def plot_seirv_model(t, y, title=None):
    S, E, I, R, V = y
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infectious')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, V, label='Vaccinated')
    plt.xlabel('Time (days)')
    plt.ylabel('Fraction of population')
    plt.legend()
    plt.grid(True)
    if title:
        plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()

if __name__ == "__main__":
    beta = 0.3
    beta_v = 0.06
    sigma = 0.2
    gamma = 0.1
    nu = 0.005
    I0 = 0.001
    V0 = 0.3
    
    t, y = run_seirv_model(beta, beta_v, sigma, gamma, nu, I0, V0)
    
    outputs = compute_outputs(t, y)
    print("Model outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value:.4f}")
    
    fig = plot_seirv_model(t, y, title="SEIRV Model with Imperfect Immunity from Vaccination")
    plt.savefig("seirv_model.png", dpi=300)
    plt.show()
