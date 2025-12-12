import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14
})

g1 = 0.70
d1 = 0.30 
k  = 1.0
g2 = 0.98
d2 = 0.70
z1 = 0.8
c1 = 0.5
c2 = 0.4
m3 = 0.2
s   = 0.4
phi = 0.2
omega1 = 0.3
z2  = 0.3
mu  = 0.29
c3    = 0.5
omega2 = 1.0
theta1 = 0.2
theta2 = 0.98
m2     = 0.10
g3 = 0.1
m1 = 0.9
A = 1.0
B = 0.1
C = 0.1

H0 = 1.0
T0 = 1e-5
I0 = 1.379310345
E0 = 0.5
F0 = 0.5
y0 = [H0, T0, I0, E0, F0]


T_end = 25
n_days = 25
t_eval = np.linspace(0, T_end, n_days + 1)

def model_with_controls(t, y, kE_schedule, kF_schedule):
    """
    Модель H,T,I,E,F з:
    - kE_schedule[day] ∈ [0,1] – блокада естрогену;
    - kF_schedule[day] ∈ [0,1] – лікування ожиріння.
    """
    H, T, I, E, F = y

    day = min(int(t), n_days - 1)
    kE = kE_schedule[day]
    kF = kF_schedule[day]
    E_eff = (1 - kE) * E
    dHdt = H * (g1 - d1 * H - k * T) - c1 * H * E_eff
    dTdt = T * (g2 - d2 * T) - z1 * I * T + c2 * H * E_eff + m3 * T * F
    dIdt = s + (phi * I * T) / (omega1 + T) - z2 * I * T - mu * I - (c3 * E_eff) / (omega2 + E_eff)
    dEdt = theta1 - theta2 * E + m2 * F * E_eff
    dFdt = (1 - kF) * g3 * F * (1 - m1 * F)

    return [dHdt, dTdt, dIdt, dEdt, dFdt]

def calculate_functional(t, T_vals, kE, kF):
    """
    J = A ∫ T(t)^2 dt + B Σ kF[i]^2 + C Σ kE[i]^2
    """
    J_tumor = np.trapz(T_vals**2, t)
    J_kE = np.sum(kE**2)
    J_kF = np.sum(kF**2)
    return A * J_tumor + B * J_kF + C * J_kE

kE = np.full(n_days, 0.3)
kF = np.full(n_days, 0.3)

tol = 1e-3
max_iter = 50
alpha = 0.3
eps = 1e-4

for it in range(max_iter):
    sol = solve_ivp(
        model_with_controls,
        [0, T_end],
        y0,
        args=(kE, kF),
        t_eval=t_eval
    )
    y = sol.y
    T_vals = y[1]
    J_base = calculate_functional(t_eval, T_vals, kE, kF)
    dkE = np.zeros_like(kE)
    dkF = np.zeros_like(kF)

    for i in range(n_days):
        kE_perturb = kE.copy()
        kE_perturb[i] += eps
        sol_perturb = solve_ivp(
            model_with_controls,
            [0, T_end],
            y0,
            args=(kE_perturb, kF),
            t_eval=t_eval
        )
        T_perturb = sol_perturb.y[1]
        J_perturb = calculate_functional(t_eval, T_perturb, kE_perturb, kF)
        dkE[i] = (J_perturb - J_base) / eps
        kF_perturb = kF.copy()
        kF_perturb[i] += eps
        sol_perturb = solve_ivp(
            model_with_controls,
            [0, T_end],
            y0,
            args=(kE, kF_perturb),
            t_eval=t_eval
        )
        T_perturb = sol_perturb.y[1]
        J_perturb = calculate_functional(t_eval, T_perturb, kE, kF_perturb)
        dkF[i] = (J_perturb - J_base) / eps
    kE_new = kE - alpha * dkE
    kF_new = kF - alpha * dkF
    kE_new = np.clip(kE_new, 0, 1)
    kF_new = np.clip(kF_new, 0, 1)
    if np.linalg.norm(kE_new - kE) + np.linalg.norm(kF_new - kF) < tol:
        kE = kE_new
        kF = kF_new
        break

    kE, kF = kE_new, kF_new

sol = solve_ivp(
    model_with_controls,
    [0, T_end],
    y0,
    args=(kE, kF),
    t_eval=t_eval
)
H, T, I, E, F = sol.y
J = calculate_functional(t_eval, T, kE, kF)

def smooth_plot(x, y, label, color):
    x_new = np.linspace(x.min(), x.max(), 500)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_new)
    plt.plot(x_new, y_smooth, label=label, linewidth=4.0, color=color)

plot_limit = t_eval <= 25
t_plot = t_eval[plot_limit]
H_plot = H[plot_limit]
T_plot = T[plot_limit]
I_plot = I[plot_limit]

plt.figure(figsize=(12, 6))
smooth_plot(t_plot, H_plot, label='H(t): нормальні клітини', color='blue')
smooth_plot(t_plot, T_plot, label='T(t): пухлинні клітини', color='red')
smooth_plot(t_plot, I_plot, label='I(t): імунні клітини', color='green')
plt.xlabel("Час", fontsize=14, fontweight='bold')
plt.ylabel("Кількість", fontsize=14, fontweight='bold')
plt.title("Динаміка клітин з оптимальним лікуванням (естроген + ожиріння)", fontsize=18, fontweight='bold')

legend = plt.legend()
for text in legend.get_texts():
    text.set_fontweight('bold')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.hlines(kE[plot_limit[:-1]], t_eval[:-1][plot_limit[:-1]], t_eval[1:][plot_limit[:-1]],
           colors='red', label='Блокада естрогену kE(t)', linewidth=3)
plt.hlines(kF[plot_limit[:-1]], t_eval[:-1][plot_limit[:-1]], t_eval[1:][plot_limit[:-1]],
           colors='purple', label='Терапія проти ожиріння kF(t)', linewidth=3)

plt.xlabel("Час", fontsize=14, fontweight='bold')
plt.ylabel("Інтенсивність лікування", fontsize=14, fontweight='bold')
plt.title("Оптимальне лікування (естроген + ожиріння)", fontsize=18, fontweight='bold')

legend = plt.legend()
for text in legend.get_texts():
    text.set_fontweight('bold')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print(f"Значення функціоналу: {J}")
print("Оптимальний профіль kE по днях:")
print(kE)
print("Оптимальний профіль kF по днях:")
print(kF)
