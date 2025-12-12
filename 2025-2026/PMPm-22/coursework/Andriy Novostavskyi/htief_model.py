import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelweight": "bold",
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16
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

def htief_model(t, y):
    H, T, I, E, F = y

    dHdt = H * (g1 - d1 * H - k * T) - c1 * H * E
    dTdt = T * (g2 - d2 * T) - z1 * I * T + c2 * H * E + m3 * T * F
    dIdt = s + (phi * I * T) / (omega1 + T) - z2 * I * T - mu * I - (c3 * E) / (omega2 + E)
    dEdt = theta1 - theta2 * E + m2 * F * E
    dFdt = g3 * F * (1 - m1 * F)

    return [dHdt, dTdt, dIdt, dEdt, dFdt]

H0 = 1.0
T0 = 1e-5
I0 = 1.379310345
E0 = 0.5
F0 = 0.5

y0 = [H0, T0, I0, E0, F0]

t_span = (0, 25)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(htief_model, t_span, y0, t_eval=t_eval, method='RK45')

H, T, I, E, F = sol.y

plt.figure(figsize=(14, 8))
plt.plot(sol.t, H, label=r'$H(t)$ — нормальні клітини', linewidth=3.5)
plt.plot(sol.t, T, label=r'$T(t)$ — пухлинні клітини', linewidth=3.5)
plt.plot(sol.t, I, label=r'$I(t)$ — імунні клітини', linewidth=3.5)
#plt.plot(sol.t, E, label=r'$E(t)$ — естроген', linewidth=3.5)
#plt.plot(sol.t, F, label=r'$F(t)$ — жирова тканина', linewidth=3.5)

plt.title('Динаміка популяцій клітин (модель з ожирінням)', fontsize=22, fontweight='bold')
plt.xlabel('Час', fontsize=18, fontweight='bold')
plt.ylabel('Відносні значення змінних', fontsize=18, fontweight='bold')

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

legend = plt.legend(loc='upper right', fontsize=15, frameon=True, fancybox=True, shadow=True)
for text in legend.get_texts():
    text.set_fontweight('bold')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
