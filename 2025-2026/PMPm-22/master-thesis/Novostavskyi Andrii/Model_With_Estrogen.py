import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def system(t, y, params):
    N, T, M, E = y

    α1, μ1, φ1, e1, α2, μ2, g, eta_g, γ1, e2, s, ρ, ω, γ2, μ3, e3, Π, θ = params

    dNdt = N * (α1 - μ1 * N - φ1 * T) - eta_g * g * N - e1 * N * E

    dTdt = T * (α2 - μ2 * T) + g * T - γ1 * M * T + e2 * N * E

    dMdt = s + (ρ * M * T) / (ω + T) - γ2 * M * T - g * M - μ3 * M - e3 * M * E

    dEdt = Π - θ * E

    return [dNdt, dTdt, dMdt, dEdt]


y0 = [1.0, 0.00001, 1.15, 0.5]

t_span = (0, 25)

t_eval = np.linspace(t_span[0], t_span[1], 1000)

α1 = 0.7
μ1 = 0.2
φ1 = 1.0
e1 = 0.4

α2 = 0.9
μ2 = 0.5
g  = 0.3
eta_g = 0.1
γ1 = 0.4
e2 = 0.3

s  = 0.5
ρ  = 0.5
ω  = 0.3
γ2 = 0.29
μ3 = 0.3
e3 = 0.3

Π  = 0.5
θ  = 1.0

params = [α1, μ1, φ1, e1, α2, μ2, g, eta_g, γ1, e2, s, ρ, ω, γ2, μ3, e3, Π, θ]

sol = solve_ivp(system, t_span, y0, args=(params,), t_eval=t_eval)

plt.figure(figsize=(9, 6))

marker_spacing = 60 

plt.plot(sol.t, sol.y[0], label=r'N(t): нормальні клітини', 
         color='blue', linestyle='-', marker='o',
         linewidth=3, markersize=8, markevery=marker_spacing)

plt.plot(sol.t, sol.y[1], label=r'T(t): пухлинні клітини', 
         color='red', linestyle='--', marker='^',
         linewidth=3, markersize=8, markevery=marker_spacing)

plt.plot(sol.t, sol.y[2], label=r'M(t): імунні клітини', 
         color='black', linestyle=':', marker='s',
         linewidth=3, markersize=8, markevery=marker_spacing)

plt.plot(sol.t, sol.y[3], label=r'E(t): естроген', 
         color='darkorange', linestyle='-.', marker='X',
         linewidth=3, markersize=8, markevery=marker_spacing)

plt.xlim(0, 25)
plt.ylim(0, 1.5)
plt.yticks([0, 0.5, 1, 1.5, 2])

plt.ylabel('Концентрація клітин', fontsize=12, fontweight='bold')
plt.title('Динаміка популяцій клітин з глюкозою та естрогеном',
          fontsize=12, fontweight='bold', pad=15)

plt.legend(loc='center right', fontsize=10,
           framealpha=1, edgecolor='gray',
           bbox_to_anchor=(0.95, 0.6))

plt.grid(True, linestyle='-', alpha=0.5)

plt.tight_layout()

plt.savefig('fig1_dynamics_model1.png', dpi=300)

plt.show()
