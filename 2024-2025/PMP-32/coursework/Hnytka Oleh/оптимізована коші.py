import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



plt.style.use('bmh') 

alpha1 = 0.0031
u = 1
def model(t, X):
    B, N, P, E = X

    s = 0.8
    L = 50
    u_t = alpha1 * (1-u)
    lambda2 = 0.0007
    pi1 = 0.03
    phi0 = 0.4
    r = 0.5
    K = 100
    pi = 0.004
    lambda_ = 0.007
    lambda0 = 0.4
    pi2 = 0.09
    gamma1 = 0.0002
    phi = 0.006

    dBdt = s * B * (1 - B / L) - u_t * B * N - lambda2 * B**2 * P + pi1 * phi0 * E
    dNdt = r * N * (1 - N / K) + pi * u_t * B * N
    dPdt = lambda_ * N - lambda0 * P - pi2 * gamma1 * P * E
    dEdt = phi * (L - B) - phi0 * E - gamma1 * P * E

    return [dBdt, dNdt, dPdt, dEdt]


X0 = [50, 30, 29, 10]


t_span = (0, 100)
t_eval_yearly = np.arange(t_span[0], t_span[1] + 1)


sol_yearly = solve_ivp(model, t_span, X0, t_eval=t_eval_yearly)

t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(model, t_span, X0, t_eval=t_eval)

fig, axs = plt.subplots(2, 2, figsize=(16, 9))
titles = ['Ліси B(t), при u(t) = 1', 'Населення N(t)', 'Тиск P(t)', 'Економічні заходи E(t)']
colors = ['forestgreen', 'royalblue', 'darkred', 'orange']
labels = ['B(t)', 'N(t)', 'P(t)', 'E(t)']

for i, ax in enumerate(axs.flat):
    ax.plot(sol.t, sol.y[i], label=labels[i], color=colors[i], linewidth=2.5,linestyle='--', alpha=0.7)
    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_xlabel('Час (роки)', fontsize=12)
    ax.set_ylabel(labels[i], fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=11)

plt.suptitle('Динаміка змін у моделі лісів, населення і тиску з керуючою функцією u(t)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
