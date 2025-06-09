import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

params = {
    'alpha1': 0.7, 'mu1': 0.3, 'phi1': 1.0,
    'alpha2': 0.5, 'mu2': 0.7, 'gamma1': 0.9,
    's': 0.4, 'rho': 0.2, 'omega': 0.3,
    'gamma2': 0.35, 'mu3': 0.29,
    'e1': 0.01, 'e2': 0.02, 'e3': 0.015,
    'Pi': 1.0, 'theta': 0.1
}
g0 = 0.46
A, B, C = 1.0, 0.4, 0.4
y0 = [1, 0.00001, 1.379310345, 0.1]
T_end = 26
n_days = 24
t_eval = np.linspace(0, T_end, n_days + 1)

def model_with_control(t, y, kG_schedule, kE_schedule):
    N, T, M, E = y
    day = min(int(t), n_days - 1)
    kG = kG_schedule[day]
    kE = kE_schedule[day]
    g = (1 - kG) * g0
    E_eff = (1 - kE) * E
    dNdt = N * (params['alpha1'] - params['mu1'] * N - params['phi1'] * T) - params['e1'] * N * E_eff
    dTdt = T * (params['alpha2'] - params['mu2'] * T) + g * T - params['gamma1'] * M * T + params['e2'] * N * E_eff
    dMdt = params['s'] + (params['rho'] * M * T) / (params['omega'] + T) - params['gamma2'] * M * T - g * M - params['mu3'] * M - params['e3'] * M * E_eff
    dEdt = params['Pi'] - params['theta'] * E
    return [dNdt, dTdt, dMdt, dEdt]

def calculate_functional(t, T_vals, kG, kE):
    J_tumor = np.trapz(T_vals**2, t)
    J_kG = np.sum(kG**2)
    J_kE = np.sum(kE**2)
    return A * J_tumor + B * J_kG + C * J_kE

kG = np.full(n_days, 0.5)
kE = np.full(n_days, 0.5)
tol = 1e-3
max_iter = 50
alpha = 0.3

for _ in range(max_iter):
    sol = solve_ivp(model_with_control, [0, T_end], y0, args=(kG, kE), t_eval=t_eval)
    y = sol.y
    T_vals = y[1]
    dkG = np.zeros_like(kG)
    dkE = np.zeros_like(kE)
    eps = 1e-4
    J_base = calculate_functional(t_eval, T_vals, kG, kE)

    for i in range(n_days):
        kG_perturb = kG.copy()
        kG_perturb[i] += eps
        sol_perturb = solve_ivp(model_with_control, [0, T_end], y0, args=(kG_perturb, kE), t_eval=t_eval)
        J_perturb = calculate_functional(t_eval, sol_perturb.y[1], kG_perturb, kE)
        dkG[i] = (J_perturb - J_base) / eps
        kE_perturb = kE.copy()
        kE_perturb[i] += eps
        sol_perturb = solve_ivp(model_with_control, [0, T_end], y0, args=(kG, kE_perturb), t_eval=t_eval)
        J_perturb = calculate_functional(t_eval, sol_perturb.y[1], kG, kE_perturb)
        dkE[i] = (J_perturb - J_base) / eps
    kG_new = np.clip(kG - alpha * dkG, 0, 1)
    kE_new = np.clip(kE - alpha * dkE, 0, 1)

    if np.linalg.norm(kG_new - kG) + np.linalg.norm(kE_new - kE) < tol:
        break
    kG, kE = kG_new, kE_new

sol = solve_ivp(model_with_control, [0, T_end], y0, args=(kG, kE), t_eval=t_eval)
N, T, M, E = sol.y
J = calculate_functional(t_eval, T, kG, kE)

def smooth_plot(x, y, label, color):
    x_new = np.linspace(x.min(), x.max(), 500)  
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_new)
    plt.plot(x_new, y_smooth, label=label, linewidth=3, color=color)

plot_limit = t_eval <= 25
t_plot = t_eval[plot_limit]
N_plot, T_plot, M_plot = N[plot_limit], T[plot_limit], M[plot_limit]

plt.figure(figsize=(10, 5))
smooth_plot(t_plot, N_plot, label='N(t): нормальні клітини', color='blue')
smooth_plot(t_plot, T_plot, label='T(t): пухлинні клітини', color='red')
smooth_plot(t_plot, M_plot, label='M(t): імунні клітини', color='green')
plt.xlabel("Час", fontsize=12, fontweight='bold')
plt.ylabel("Кількість / Рівень", fontsize=12, fontweight='bold')
plt.title("Динаміка клітин з оптимальним лікуванням", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.hlines(kG[plot_limit[:-1]], t_eval[:-1][plot_limit[:-1]], t_eval[1:][plot_limit[:-1]], colors='blue', label='Інгібування глюкози (kG)', linewidth=2)
plt.hlines(kE[plot_limit[:-1]], t_eval[:-1][plot_limit[:-1]], t_eval[1:][plot_limit[:-1]], colors='red', label='Блокада естрогену (kE)', linewidth=2)
plt.xlabel("Час", fontsize=12, fontweight='bold')
plt.ylabel("Інтенсивність лікування", fontsize=12, fontweight='bold')
plt.title("Оптимальне дискретне лікування", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Значення функціоналу: {J}")
