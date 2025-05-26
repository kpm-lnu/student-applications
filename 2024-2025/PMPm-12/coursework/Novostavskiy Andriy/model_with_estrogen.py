import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def system(t, y, params):
    N, T, M, E = y
    α1, μ1, φ1, e1, α2, μ2, g, γ1, e2, s, ρ, ω, γ2, μ3, e3, Π, θ = params
    dNdt = N * (α1 - μ1 * N - φ1 * T) - e1 * N * E
    dTdt = T * (α2 - μ2 * T) + g * T - γ1 * M * T + e2 * N * E
    dMdt = s + (ρ * M * T) / (ω + T) - γ2 * M * T - g * M - μ3 * M - e3 * M * E
    dEdt = Π - θ * E

    return [dNdt, dTdt, dMdt, dEdt]

y0 = [1, 0.00001, 1.379310345, 0.1]  
t_span = (0, 25)
t_eval = np.linspace(*t_span, 1000)
α1 = 0.7
μ1 = 0.3
φ1 = 1
e1 = 0.5
α2 = 0.9
μ2 = 0.7
g = 0.3
γ1 = 0.4
e2 = 0.4
s = 0.4
ρ = 0.3
ω = 0.3
γ2 = 0.29
μ3 = 0.5
e3 = 1
Π = 0.1
θ = 1
params = [α1, μ1, φ1, e1, α2, μ2, g, γ1, e2, s, ρ, ω, γ2, μ3, e3, Π, θ]
sol = solve_ivp(system, t_span, y0, args=(params,), t_eval=t_eval)

plt.figure(figsize=(12, 7))
plt.plot(sol.t, sol.y[0],label=r'$N(t)\colon$ нормальні клітини', linewidth=4)
plt.plot(sol.t, sol.y[1], label=r'$T(t)\colon$ пухлинні клітини', linewidth=4)
plt.plot(sol.t, sol.y[2], label=r'$M(t)\colon$ імунні клітини', linewidth=4)
plt.xlabel('Час', fontsize=16, fontweight='bold')
plt.ylabel('Кількість клітин', fontsize=16, fontweight='bold')
plt.title('Динаміка популцій клітин з глюкозою та естрогеном', fontsize=18, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()
