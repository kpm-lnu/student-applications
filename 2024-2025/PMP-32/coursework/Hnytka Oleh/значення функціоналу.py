import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt


params = {
    's': 0.8, 'L': 50, 'alpha0': 0.0031, 'lambda2': 0.0007,
    'pi1': 0.03, 'phi0': 0.4, 'r': 0.5, 'K': 100,
    'pi': 0.004, 'lambda_': 0.007, 'lambda0': 0.4,
    'pi2': 0.09, 'gamma1': 0.0002, 'phi': 0.006
}

weights = {
    'ω1': 1.0,
    'ω2': 0.5  
}

def model(t, X, u):

    B, N, P, E = X
    alpha1 = params['alpha0'] * (1 - u)
    
    dBdt = (params['s'] * B * (1 - B/params['L']) - alpha1 * B * N - 
            params['lambda2'] * B**2 * P + params['pi1'] * params['phi0'] * E)
    
    dNdt = (params['r'] * N * (1 - N/params['K']) + params['pi'] * alpha1 * B * N)
    
    dPdt = params['lambda_'] * N - params['lambda0'] * P - params['pi2'] * params['gamma1'] * P * E
    
    dEdt = params['phi'] * (params['L'] - B) - params['phi0'] * E - params['gamma1'] * P * E
    
    return [dBdt, dNdt, dPdt, dEdt]

def compute_functional(u_value, t_span=(0, 100), t_eval=None):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    sol = solve_ivp(model, t_span, [50, 30, 29, 10], 
                   args=(u_value,), t_eval=t_eval)
    
    B = sol.y[0] 
    u_squared = u_value**2 * np.ones_like(sol.t)
    
    integrand = weights['ω1'] * B - weights['ω2'] * u_squared
    
    J = trapezoid(integrand, sol.t)
    
    return J, sol

u_values = [0.2, 0.5, 0.8]
results = {}

for u in u_values:
    J, sol = compute_functional(u)
    results[u] = {
        'J': J,
        'solution': sol
    }
    print(f"For u = {u}: J(u) = {J:.2f}")

plt.figure(figsize=(12, 6))
for u, data in results.items():
    plt.plot(data['solution'].t, data['solution'].y[0], 
             label=f'B(t), u={u} (J={data["J"]:.2f})')
plt.title('Forest dynamics B(t) for different control levels')
plt.xlabel('Time (years)')
plt.ylabel('Forest area B(t)')
plt.legend()
plt.grid(True)
plt.show()