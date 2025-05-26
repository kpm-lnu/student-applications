import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.random.seed(42)

params = {
    'gamma_h': 0.04,
    'lambda_hI': 0.025,
    'r1': 0.005,
    'lambda_hS': 0.0001,
    'r2': 0.0025,
    'mu_rI': 0.02,
    'lambda_rI': 0.04,
    'gamma_r': 0.05,
    'theta': 0.5,
    'delta': 0.1,
    'alpha': 0.2
}

def model(t, y, u_func, params):
    hI, hR, rI, E = y
    u = u_func(t)
    
    dhI = params['gamma_h'] * (1 - u) * rI * (1 - hI - hR) + params['theta'] * E * (1 - hI - hR) - hI * (params['lambda_hI'] + params['r1'])
    dhR = params['r1'] * hI - hR * (params['lambda_hS'] + params['r2'])
    drI = (params['mu_rI']*rI - params['lambda_rI']) * rI + params['gamma_r'] * rI * (1 - rI) * (1 - u)
    dE = params['alpha'] * rI - params['delta'] * E
    return [dhI, dhR, drI, dE]

def cost_function(y, u):
    w2, cu =  1.0, 1.5
    return np.trapz(w2*y[:,2] + cu*u**2, dx=dt)

T = 2000
N = 20
dt = T/N
y0 = [0.0, 0.0, 0.1, 0.0]
time_points = np.linspace(0, T, N+1)

control_start_day = 200
control_start_index = int(control_start_day / dt)

samples = 1000

U_data = np.random.uniform(0, 0.3, (samples, N))
U_data[:, :control_start_index] = 0 

def simulate_controlled(u):
    y_prev = y0
    y_full = []
    for i in range(N):
        sol = solve_ivp(model, [0, dt], y_prev, args=(lambda t: u[i], params), t_eval=[dt])
        y_full.append(sol.y[:,-1])
        y_prev = sol.y[:,-1]
    return np.array(y_full)

Y_data = np.array([simulate_controlled(u) for u in U_data])
C_data = np.array([cost_function(Y_data[i], U_data[i]) for i in range(samples)])

kernel = RBF(length_scale=0.5, length_scale_bounds=(1e-2, 10.0))
gp = BaggingRegressor(
        estimator=GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        n_restarts_optimizer=10
    ),
    n_estimators=10
)
X_train, X_test, y_train, y_test = train_test_split(U_data, C_data, test_size=0.2, random_state=42)
gp.fit(X_train, y_train)

def smoothness_constraint(u):
    return np.sum(np.diff(u)**2)

def surrogate_cost(u_var):
    u = np.concatenate([np.zeros(control_start_index), u_var])
    u = u.reshape(1, -1)
    cost = gp.predict(u)[0]
    cost += 0.001 * smoothness_constraint(u[0])  
    cost += 0.01 * np.mean((u - 0.15)**2)        
    return cost

bounds = [(0.01, 0.3)] * (N - control_start_index)
res = minimize(
    surrogate_cost,
    x0=np.linspace(0.01, 0.3, N - control_start_index),
    bounds=bounds,
    method='SLSQP',
    options={'maxiter': 1000, 'ftol': 1e-6}
)

u_opt_full = np.concatenate([np.zeros(control_start_index), res.x])

u_func = interp1d(
    np.append(time_points[:-1], time_points[-1]),
    np.append(u_opt_full, u_opt_full[-1]),
    kind='cubic',
    fill_value="extrapolate"
)

sol = solve_ivp(model, [0, T], y0, args=(u_func, params), t_eval=np.linspace(0, T, 1000))
t_result, y_result = sol.t, sol.y

plt.figure(figsize=(12, 6))
plt.plot(t_result, y_result[0], label='Інфіковані люди $h_I$')
plt.plot(t_result, y_result[1], label='Вилікувані люди $h_R$')
plt.plot(t_result, y_result[2], label='Інфіковані переносники $r_I$')
plt.plot(t_result, y_result[3], label='Забруднення $E$')
plt.xlabel('Час (дні)')
plt.ylabel('Частка популяції')
plt.legend()
plt.grid(True)
plt.title('Динаміка системи з оптимальним керуванням (з 100 дня)')
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(time_points[:-1], u_opt_full, 'o', label='Оптимальні точки', markersize=6)
plt.plot(np.linspace(0, T, 1000), u_func(np.linspace(0, T, 1000)), '-', label='Інтерпольоване керування')
plt.xlabel('Час (дні)')
plt.ylabel('Інтенсивність вилову $u(t)$')
plt.legend()
plt.grid(True)
plt.title('Оптимальна стратегія вилову гризунів (з 100 дня)')
plt.show()
