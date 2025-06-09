import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "legend.fontsize": 14
})

alpha1 = 0.7
mu1 = 0.3
phi1 = 1.0
alpha2 = 0.5
mu2 = 0.7
gamma1 = 0.9
rho = 0.2
omega = 0.3
gamma2 = 0.35
mu3 = 0.29
g = 0.3
phi2 = 0.8
s = 0.4

def model(t, y, s):
    N, T, M = y
    dNdt = N * (alpha1 - mu1 * N - phi1 * T)
    dTdt = T * (alpha2 - mu2 * T + g - gamma1 * M + phi2 * N)
    dMdt = s + (rho * M * T) / (omega + T) - gamma2 * M * T - mu3 * M - g * M

    return [dNdt, dTdt, dMdt]

y0 = [1, 0.00001, 1.379310345]
t_span = (0, 30)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(model, t_span, y0, args=(s,), t_eval=t_eval, method='RK45')
N, T, M = sol.y

plt.figure(figsize=(12, 7))
plt.plot(sol.t, N, label=r'$N(t)\colon$ нормальні клітини', linewidth=4)
plt.plot(sol.t, T, label=r'$T(t)\colon$ пухлинні клітини', linewidth=4)
plt.plot(sol.t, M, label=r'$M(t)\colon$ імунні клітини', linewidth=4)
plt.title('Динаміка популяцій клітин', fontsize=18, fontweight='bold')
plt.xlabel('Час', fontsize=16, fontweight='bold')
plt.ylabel('Кількість клітин', fontsize=16, fontweight='bold')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(loc='upper right', fontsize=18, frameon=True, fancybox=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()