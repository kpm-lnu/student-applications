import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def model_with_estrogen(t, y, params):
    N, T, M, E = y
    
    alpha1, mu1, phi1 = params['alpha1'], params['mu1'], params['phi1']
    alpha2, mu2, gamma1, g = params['alpha2'], params['mu2'], params['gamma1'], params['g']
    eta_g = params['eta_g']
    s, rho, omega, gamma2, mu3 = params['s'], params['rho'], params['omega'], params['gamma2'], params['mu3']
    e1, e2, e3, Pi, theta = params['e1'], params['e2'], params['e3'], params['Pi'], params['theta']

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


initial_conditions = [1.0, 0.00001, 1.15, 0.5]

t_span = (0, 25)

t_eval = np.linspace(t_span[0], t_span[1], 1000)


base_params = {
    'alpha1': 0.7, 'mu1': 0.2, 'phi1': 1.0,
    'alpha2': 0.9, 'mu2': 0.5, 'gamma1': 0.4, 'g': 0.3, 'eta_g': 0.1,
    's': 0.5, 'rho': 0.5, 'omega': 0.3, 'gamma2': 0.29, 'mu3': 0.3,
    'e1': 0.4, 'e2': 0.3, 'e3': 0.3, 'Pi': 0.5, 'theta': 1.0
}


glucose_levels = {
    'g = 0.01': 0.01,
    'g = 0.2': 0.2,
    'g = 0.3': 0.3,
    'g = 0.46': 0.46,
    'g = 0.8': 0.8,
}


styles = ['-o', '--^', '-.s', ':D', '-X']
colors = ['blue', 'green', 'orange', 'red', 'purple']
marker_spacing = 80


plt.figure(figsize=(10, 6))

for (label, g_val), style, color in zip(glucose_levels.items(), styles, colors):
    params = base_params.copy()
    params['g'] = g_val

    sol = solve_ivp(
        model_with_estrogen,
        t_span,
        initial_conditions,
        args=(params,),
        t_eval=t_eval
    )

    plt.plot(
        sol.t,
        sol.y[0],
        style,
        color=color,
        label=rf'N(t), {label}',
        linewidth=3,
        markersize=8,
        markevery=marker_spacing
    )

plt.title('Динаміка нормальних клітин при різних рівнях глюкози',
          fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Час', fontsize=14, fontweight='bold')
plt.ylabel('Концентрація нормальних клітин', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('fig3_glucose_dependence_c.png', dpi=300)
plt.show()


plt.figure(figsize=(10, 6))

for (label, g_val), style, color in zip(glucose_levels.items(), styles, colors):
    params = base_params.copy()
    params['g'] = g_val

    sol = solve_ivp(
        model_with_estrogen,
        t_span,
        initial_conditions,
        args=(params,),
        t_eval=t_eval
    )

    plt.plot(
        sol.t,
        sol.y[1],
        style,
        color=color,
        label=rf'$T(t)$, {label}',
        linewidth=3,
        markersize=8,
        markevery=marker_spacing
    )

plt.title('Динаміка пухлинних клітин при різних рівнях глюкози',
          fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Час', fontsize=14, fontweight='bold')
plt.ylabel('Концентрація пухлинних клітин', fontsize=14, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('fig3_glucose_dependence_a.png', dpi=300)
plt.show()


plt.figure(figsize=(10, 6))

for (label, g_val), style, color in zip(glucose_levels.items(), styles, colors):
    params = base_params.copy()
    params['g'] = g_val

    sol = solve_ivp(
        model_with_estrogen,
        t_span,
        initial_conditions,
        args=(params,),
        t_eval=t_eval
    )

    plt.plot(
        sol.t,
        sol.y[2],
        style,
        color=color,
        label=rf'$M(t)$, {label}',
        linewidth=3,
        markersize=8,
        markevery=marker_spacing
    )

plt.title('Динаміка імунних клітин при різних рівнях глюкози',
          fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Час', fontsize=14, fontweight='bold')
plt.ylabel('Концентрація імунних клітин', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('fig3_glucose_dependence_b.png', dpi=300)
plt.show()
