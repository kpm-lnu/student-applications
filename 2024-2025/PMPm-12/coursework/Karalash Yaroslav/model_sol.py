import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Параметри ---
γh = 0.04
λhI = 0.025
r1 = 0.005
λhS = 0.0001
r2 = 0.0025
μrI = 0.02
λrI = 0.04
γr = 0.05
α = 0.5
η = 0.2
δ = 0.1

def leptospirosis_model(t, y):
    Ih, Rh, Ir, E = y

    dIh_dt = γh * (Ir + α * E) * (1 - Ih - Rh) - (λhI + r1) * Ih
    dRh_dt = r1 * Ih - (λhS + r2) * Rh
    dIr_dt = μrI * Ir - λrI * Ir + γr * Ir * (1 - Ir)
    dE_dt = η * Ir - δ * E

    return [dIh_dt, dRh_dt, dIr_dt, dE_dt]

y0 = [0.0, 0.0, 0.1, 0.0]  # Ih, Rh, Ir, E

T = 2000
t_eval = np.linspace(0, T, 2000)

solution = solve_ivp(leptospirosis_model, [0, T], y0, t_eval=t_eval)

plt.figure(figsize=(12, 8))
plt.plot(solution.t, solution.y[0], label='Інфіковані люди Ih(t)', color='red')
plt.plot(solution.t, solution.y[1], label='Одужалі люди Rh(t)', color='green')
plt.plot(solution.t, solution.y[2], label='Інфіковані переносники Ir(t)', color='blue')
plt.plot(solution.t, solution.y[3], label='Забруднення середовища E(t)', color='orange')

plt.xlabel('Час (дні)')
plt.ylabel('Частка / Рівень')
plt.title('Динаміка поширення лептоспірозу з урахуванням середовища')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
