import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "legend.fontsize": 14
})

alpha1 = 0.7
mu1 = 0.2
phi1 = 1.0

alpha2 = 0.9
mu2 = 0.5
gamma1 = 0.4

eta_g = 0.1

s = 0.5
rho = 0.5
omega = 0.3
gamma2 = 0.29
mu3 = 0.3

e1 = 0.4
e2 = 0.3
e3 = 0.3
Pi = 0.5
theta = 1.0

y0 = [1.0, 0.00001, 1.15, 0.5]

t_span = (0, 25)
t_eval = np.linspace(*t_span, 1000)

g_values = [
    (0.46, "Рівень глюкози"),
    (0.3,  "Рівень глюкози"),
    (0.2,  "Рівень глюкози"),
    (0.01, "Рівень глюкози")
]

def full_model(t, y, g):
    N, T, M, E = y
    
    dNdt = N * (alpha1 - mu1 * N - phi1 * T) \
           - eta_g * g * N \
           - e1 * N * E

    dTdt = T * (alpha2 - mu2 * T) \
           + g * T \
           - gamma1 * M * T \
           + e2 * N * E

    dMdt = s \
           + (rho * M * T) / (omega + T) \
           - gamma2 * M * T \
           - g * M \
           - mu3 * M \
           - e3 * M * E

    dEdt = Pi - theta * E
    
    return [dNdt, dTdt, dMdt, dEdt]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

lines = []
labels = [
    r'N(t): нормальні клітини',
    r'T(t): пухлинні клітини',
    r'M(t): імунні клітини',
    r'E(t): естроген'
]

for ax, (g_val, label) in zip(axes, g_values):

    sol = solve_ivp(
        full_model,
        t_span,
        y0,
        args=(g_val,),
        t_eval=t_eval,
        method='RK45'
    )

    N, T, M, E = sol.y
    
    line_n, = ax.plot(sol.t, N, color='blue', linestyle='-', linewidth=3)
    line_t, = ax.plot(sol.t, T, color='red', linestyle='--', linewidth=3)
    line_m, = ax.plot(sol.t, M, color='black', linestyle=':', linewidth=3)
    line_e, = ax.plot(sol.t, E, color='darkorange', linestyle='-.', linewidth=3)

    ax.set_title(f'{label} ($g = {g_val}$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Час', fontsize=12, fontweight='bold')
    ax.set_ylabel('Концентрація клітин', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 2.5)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if len(lines) == 0:
        lines = [line_n, line_t, line_m, line_e]

plt.tight_layout()

plt.subplots_adjust(bottom=0.12)

fig.legend(
    lines,
    labels,
    loc='lower center',
    ncol=4,
    fontsize=14,
    bbox_to_anchor=(0.5, 0.01),
    frameon=True,
    shadow=True
)

plt.savefig('fig2_difference_g.png', dpi=300, bbox_inches='tight')

plt.show()
