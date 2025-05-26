import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.font_manager import FontProperties

def system(t, y, control):
    H, T, I, E = y
    alpha1, beta1, delta1, sigma1 = 0.7, 0.3, 1, 0.5
    alpha3, beta2, gamma2, sigma2 = 0.98, 0.7, 0.8, 0.4
    s, rho, omega, gamma3, mu, sigma3, v = 0.4, 0.2, 0.3, 0.3, 0.29, 0.5, 1
    Pi, theta = 0.5, 1

    if isinstance(control, (np.ndarray, list)):
        u = control[int(t)] if int(t) < len(control) else control[-1]
    else:
        u = control

    dH_dt = H * (alpha1 - beta1 * H - delta1 * T) - (1 - u) * sigma1 * H * E
    dT_dt = T * (alpha3 - beta2 * T) - gamma2 * I * T + (1 - u) * sigma2 * H * E
    dI_dt = s + (rho * I * T) / (omega + T) - gamma3 * I * T - mu * I - (1 - u) * (sigma3 * I * E) / (v + E)
    dE_dt = (1 - u) * Pi - theta * E

    return [dH_dt, dT_dt, dI_dt, dE_dt]

def objective(control, B):
    control = np.clip(control, 0, 1)  # Ensure control is within [0, 1]
    sol = solve_ivp(lambda t, y: system(t, y, control), t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], len(control)))
    H, T, I, E = sol.y
    dt = (t_span[1] - t_span[0]) / len(control)  # Time step
    state_cost = np.sum(T) * dt  # Integral approximation for T(t)
    control_cost = np.sum(0.5 * B * control**2) * dt  # Integral approximation for 0.5 * B * u(t)^2
    cost = state_cost + control_cost

    return cost

y0 = [1, 0.00001, 1.379310345, 0.5]
t_span = (0, 25)
control_points = 25
B = 0.01
np.random.seed(50)

initial_guess = np.random.uniform(0, 1, control_points)
initial_guess[0] = 0

bounds = [(0, 1)] * control_points
bounds[0] = (0, 0)

result = minimize(objective, initial_guess, args=(B,), method='L-BFGS-B', bounds=bounds)

optimal_control = result.x
print("Optimal control vector:", optimal_control)
print("Objective function value:", result.fun)

sol = solve_ivp(lambda t, y: system(t, y, optimal_control), t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], len(optimal_control)))

H, T, I, E = sol.y
time = sol.t

H_initial, T_initial, I_initial, E_initial = H[0], T[0], I[0], E[0]
H_smooth = gaussian_filter1d(H, sigma=1)
T_smooth = gaussian_filter1d(T, sigma=1)
I_smooth = gaussian_filter1d(I, sigma=1)
E_smooth = gaussian_filter1d(E, sigma=1)

H_smooth[0], T_smooth[0], I_smooth[0], E_smooth[0] = H_initial, T_initial, I_initial, E_initial

plt.figure(figsize=(10, 6))
plt.plot(time, H_smooth, label='Normal Cells (H)', color='blue', linewidth=3)
plt.plot(time, T_smooth, label='Tumor Cells (T)', color='red', linewidth=3)
plt.plot(time, I_smooth, label='Immune Cells (I)', color='green', linewidth=3)
#plt.plot(time, E_smooth, label='Estrogen (E)', color='red')
plt.xlabel('Час',fontdict={'fontsize': 16, 'weight': 'bold'})
plt.ylabel('Популяції',fontdict={'fontsize': 16, 'weight': 'bold'})
legend_properties = FontProperties(size=16, weight='bold')
plt.legend((r'$H(t)$', r'$T(t)$', r'$I(t)$'), loc='upper right', bbox_to_anchor=(1.1, 1), prop=legend_properties)
plt.title('Динаміка клітинних популяцій з оптимальним контролем',fontdict={'fontsize': 16, 'weight': 'bold'})

plt.grid(True)
plt.show()

# Plot the control function u(t)
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
plt.plot(np.linspace(t_span[0], t_span[1], len(optimal_control)), optimal_control, label='Control Function (u)', color='black', linewidth=2)
plt.xlabel('Час',fontdict={'fontsize': 16, 'weight': 'bold'})
plt.ylabel('Сила боротьби',fontdict={'fontsize': 16, 'weight': 'bold'})
plt.title('Оптимальна функція контролю за часом',fontdict={'fontsize': 16, 'weight': 'bold'})
plt.legend()
plt.grid(True)
plt.show()
