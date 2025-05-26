import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd


def model(t, X):
    B, N, P, E = X

    s = 1.6
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
    phi = 0.006

    dBdt = s * B * (1 - B / L) - alpha1 * B * N - lambda2 * B**2 * P + pi1 * phi0 * E
    dNdt = r * N * (1 - N / K) + pi * alpha1 * B * N
    dPdt = lambda_ * N - lambda0 * P - pi2 * gamma1 * P * E
    dEdt = phi * (L - B) - phi0 * E - gamma1 * P * E

    return [dBdt, dNdt, dPdt, dEdt]


X0 = [50, 30, 29, 10]


t_span = (0, 100)
t_eval_yearly = np.arange(t_span[0], t_span[1] + 1)


sol_yearly = solve_ivp(model, t_span, X0, t_eval=t_eval_yearly)
df_yearly = pd.DataFrame({
    'Рік': sol_yearly.t,
    'Ліси B(t)': sol_yearly.y[0],
    'Населення N(t)': sol_yearly.y[1],
    'Тиск P(t)': sol_yearly.y[2],
    'Економічні заходи E(t)': sol_yearly.y[3]
})
df_yearly.to_excel('model_results.xlsx', index=False)

t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(model, t_span, X0, t_eval=t_eval)

fig, axs = plt.subplots(2, 2, figsize=(14, 8))  # зручний розмір

titles = ['Ліси B(t)', 'Населення N(t)', 'Тиск P(t)', 'Економічні заходи E(t)']
colors = ['green', 'blue', 'red', 'orange']

for i, ax in enumerate(axs.flat):
    ax.plot(sol.t, sol.y[i], color=colors[i])
    ax.set_title(titles[i], fontsize=12)
    ax.set_xlabel('Час')
    ax.set_ylabel('Значення')
    ax.grid(True)  # ← сітка увімкнена

plt.tight_layout()
plt.show()



# Виведення наближеної нетривіальної точки рівноваги (остання точка розв'язку)
B_star = sol.y[0, -1]
N_star = sol.y[1, -1]
P_star = sol.y[2, -1]
E_star = sol.y[3, -1]

print("Наближена нетривіальна точка рівноваги на момент часу t = 100:")
print(f"B* ≈ {B_star:.5f}")
print(f"N* ≈ {N_star:.5f}")
print(f"P* ≈ {P_star:.5f}")
print(f"E* ≈ {E_star:.5f}")
