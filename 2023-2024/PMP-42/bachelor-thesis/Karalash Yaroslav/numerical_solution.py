import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Параметри моделі
gamma_h = 0.04
lambda_hI = 0.025
r1 = 0.005
lambda_hS = 0.0001
r2 = 0.0025
mu_rI = 0.02
lambda_rI = 0.04
gamma_r = 0.05

# Початкові умови (наприклад, 1% інфікованих людей та гризунів)
I_h0 = 0.0
R_h0 = 0.0
I_r0 = 0.1
initial_conditions = [I_h0, R_h0, I_r0]

# Часовий діапазон (наприклад, 0-365 днів)
t = np.linspace(0, 2000, 1000)

# Система диференціальних рівнянь
def model(y, t, gamma_h, lambda_hI, r1, lambda_hS, r2, mu_rI, lambda_rI, gamma_r):
    I_h, R_h, I_r = y
    
    dI_h_dt = gamma_h * I_r * (1 - I_h - R_h) - I_h * (lambda_hI + r1)
    dR_h_dt = r1 * I_h - R_h * (lambda_hS + r2)
    dI_r_dt = mu_rI * I_r - lambda_rI * I_r + gamma_r * I_r * (1 - I_r)
    
    return [dI_h_dt, dR_h_dt, dI_r_dt]

# Розв'язок системи диференціальних рівнянь
solution = odeint(model, initial_conditions, t, args=(gamma_h, lambda_hI, r1, lambda_hS, r2, mu_rI, lambda_rI, gamma_r))
I_h, R_h, I_r = solution.T

# Побудова графіків
plt.figure(figsize=(12, 8))
plt.plot(t, I_h, label='Інфіковані люди (I_h)')
plt.plot(t, R_h, label='Одужалі люди (R_h)')
plt.plot(t, I_r, label='Інфіковані гризуни (I_r)')
plt.xlabel('Час (дні)')
plt.ylabel('Частка популяції')
plt.title('Динаміка поширення хантавірусу')
plt.legend()
plt.grid(True)
plt.show()
