import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def model(t, X, alpha1):
    B, N, P, E = X

    s = 0.8
    L = 50
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

    dBdt = s*B*(1 - B/L) - alpha1*B*N - lambda2*B**2*P + pi1*phi0*E
    dNdt = r*N*(1 - N/K) + pi*alpha1*B*N
    dPdt = lambda_*N - lambda0*P - pi2*gamma1*P*E
    dEdt = phi*(L - B) - phi0*E - gamma1*P*E

    return [dBdt, dNdt, dPdt, dEdt]


X0 = [0.3, 0.3, 0.2, 0.5]
t_span = (0, 100)
t_eval = np.arange(t_span[0], t_span[1] + 0.1, 0.1)

alpha1_values = [0.0015, 0.0031, 0.006]
colors = ['blue', 'green', 'red']

results = []

for alpha1 in alpha1_values:
    sol = solve_ivp(lambda t, X: model(t, X, alpha1), t_span, X0, t_eval=t_eval)
    results.append((alpha1, sol))

variables = ['Ліси B(t)', 'Населення N(t)', 'Тиск P(t)', 'Економічні заходи E(t)']

plt.figure(figsize=(16, 10))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    for idx, (alpha1, sol) in enumerate(results):
        plt.plot(sol.t, sol.y[i], label=f'α₁ = {alpha1}', color=colors[idx])
    plt.title(variables[i])
    plt.xlabel('Час')
    plt.ylabel(variables[i].split()[0])
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
