import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

plt.style.use('bmh') 

def model(t, X, phi):
    B, N, P, E = X

    s = 0.8
    L = 50
    alpha1 = 0.0031
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

    dBdt = s * B * (1 - B / L) - alpha1 * B * N - lambda2 * B**2 * P + pi1 * phi0 * E
    dNdt = r * N * (1 - N / K) + pi * alpha1 * B * N
    dPdt = lambda_ * N - lambda0 * P - pi2 * gamma1 * P * E
    dEdt = phi * (L - B) - phi0 * E - gamma1 * P * E

    return [dBdt, dNdt, dPdt, dEdt]

phi_val = [0.003, 0.006, 0.009]
X0 = [28.4408, 100.0705, 1.7512, 0.3231]
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 1500)

fig, axs = plt.subplots(2, 2, figsize=(16, 9))
titles = ['Ліси B(t)', 'Населення N(t)', 'Тиск P(t)', 'Економічні заходи E(t)']
labels = ['B(t)', 'N(t)', 'P(t)', 'E(t)']

for phi in phi_val:
    sol = solve_ivp(model, t_span, X0, args=(phi,), t_eval=t_eval, method='LSODA')
    for i, ax in enumerate(axs.flat):
        ax.plot(sol.t, sol.y[i], label=f'φ = {phi}', linewidth=2.5)

for i, ax in enumerate(axs.flat):
    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_xlabel('Час (роки)', fontsize=12)
    ax.set_ylabel(labels[i], fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.7f'))

plt.suptitle('Динаміка змін у моделі лісів, населення і тиску при різних φ', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
