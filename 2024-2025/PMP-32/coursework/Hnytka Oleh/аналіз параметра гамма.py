import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def model(y, t, params):
    B, N, P, E = y
    s, L, α1, λ2, π1, φ0, r, K, π, λ, λ0, π2, γ, φ = params
    dBdt = s * B * (1 - B / L) - α1 * B * N - λ2 * B**2 * P + π1 * φ0 * E
    dNdt = r * N * (1 - N / K) + π * α1 * B * N
    dPdt = λ * N - λ0 * P - π2 * γ * P * E
    dEdt = φ * (L - B) - φ0 * E - γ * P * E
    return [dBdt, dNdt, dPdt, dEdt]

y0 = [28.4408, 100.0705, 1.7512, 0.3231]

s = 0.8
L = 50
α1 = 0.0031
λ2 = 0.0007
π1 = 0.03
φ0 = 0.4
r = 0.5
K = 100
π = 0.004
λ = 0.007
λ0 = 0.4
π2 = 0.09
φ = 0.006

gamma_values = [0.0001, 0.0002, 0.0003] 
t = np.linspace(0, 100, 1000)

variable_names = ['B (ліси)', 'N (населення)', 'P (тиск)', 'E (економ. заходи)']
colors = ['blue', 'green', 'red']

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i in range(4):
    for j, γ in enumerate(gamma_values):
        params = [s, L, α1, λ2, π1, φ0, r, K, π, λ, λ0, π2, γ, φ]
        sol = odeint(model, y0, t, args=(params,))
        axs[i].plot(t, sol[:, i], label=f'γ = {γ}', color=colors[j])
    
    axs[i].set_title(variable_names[i])
    axs[i].set_xlabel("Час t")
    axs[i].set_ylabel(variable_names[i])
    axs[i].legend()

plt.tight_layout()
plt.suptitle("Залежність змінних від часу при різних γ", fontsize=16, y=1.03)
plt.show()
