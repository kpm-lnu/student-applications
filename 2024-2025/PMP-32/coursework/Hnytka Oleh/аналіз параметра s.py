import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def model(y, t, params):
    B, N, P, E = y
    s, L, α1, λ2, π1, φ0, r, K, π, λ, λ0, π2, γ1, φ = params
    dBdt = s * B * (1 - B / L) - α1 * B * N - λ2 * B**2 * P + π1 * φ0 * E
    dNdt = r * N * (1 - N / K) + π * α1 * B * N
    dPdt = λ * N - λ0 * P - π2 * γ1 * P * E
    dEdt = φ * (L - B) - φ0 * E - γ1 * P * E
    return [dBdt, dNdt, dPdt, dEdt]

y0 = [20, 50, 0.874982, 0.449803]

L = 50
π = 0.004
π1 = 0.03
π2 = 0.09
r = 0.5
K = 100
λ = 0.007
λ0 = 0.4
λ2 = 0.0007
φ = 0.006
φ0 = 0.4
γ1 = 0.0002
α1 = 0.0031

t = np.linspace(0, 100, 1000)

s_values = [0.4, 0.8, 1.2]
colors = ['red', 'orange', 'green', 'blue']
labels = [f's = {val}' for val in s_values]

fig, axs = plt.subplots(2, 2, figsize=(14, 8))

for s, color, label in zip(s_values, colors, labels):
    params = [s, L, α1, λ2, π1, φ0, r, K, π, λ, λ0, π2, γ1, φ]
    sol = odeint(model, y0, t, args=(params,))
    B, N, P, E = sol.T

    axs[0, 0].plot(t, B, label=label, color=color)
    axs[0, 1].plot(t, N, label=label, color=color)
    axs[1, 0].plot(t, P, label=label, color=color)
    axs[1, 1].plot(t, E, label=label, color=color)

axs[0, 0].set_title('B (лісові ресурси)')
axs[0, 0].set_xlabel('Час t')
axs[0, 0].set_ylabel('B(t)')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].set_title('N (населення)')
axs[0, 1].set_xlabel('Час t')
axs[0, 1].set_ylabel('N(t)')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].set_title('P (тиск)')
axs[1, 0].set_xlabel('Час t')
axs[1, 0].set_ylabel('P(t)')
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].set_title('E (економічні зусилля)')
axs[1, 1].set_xlabel('Час t')
axs[1, 1].set_ylabel('E(t)')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
