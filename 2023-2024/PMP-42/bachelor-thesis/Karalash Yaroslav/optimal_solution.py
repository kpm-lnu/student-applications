import numpy as np
from scipy.integrate import solve_ivp
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
A = 1  # вага I_h
B = 1 # вага u^2

# Початкові умови
I_h0, R_h0, I_r0 = 0, 0, 0.1  # Початкові популяції (нормалізовані)

# Початкові умови для спряжених змінних
lambda_10, lambda_20, lambda_30 = 0.0, 0.0, 0.1

# Визначення системи диференціальних рівнянь з контролем
def model(t, y):
    I_h, R_h, I_r, lambda_1, lambda_2, lambda_3 = y
    u = I_r * (-I_h * gamma_h * lambda_1 - I_r * gamma_r * lambda_3 - R_h * gamma_h * lambda_1 + gamma_h * lambda_1 + gamma_r * lambda_3) / (2 * B)
    print(u)
    u = np.clip(u, 0, 0.5)  # обмежуємо значення u в межах [0, 1]
    dI_h_dt = gamma_h * (1 - u) * I_r * (1 - I_h - R_h) - I_h * (lambda_hI + r1)
    dR_h_dt = r1 * I_h - R_h * (lambda_hS + r2)
    dI_r_dt = mu_rI * I_r - lambda_rI * I_r + gamma_r * (1 - u) * I_r * (1 - I_r)
    dlambda_1_dt = -A + lambda_1 * (-gamma_h * (1 - u) * I_r - (lambda_hI + r1)) - lambda_2 * r1
    dlambda_2_dt = lambda_1 * gamma_h * (1 - u) * I_r - lambda_2 * (-lambda_hS - r2)
    dlambda_3_dt = -lambda_1 * gamma_h * (1 - u) * (1 - I_h - R_h) + lambda_3 * (lambda_rI - mu_rI + gamma_r * (1 - u) * (1 - 2 * I_r))
    return [dI_h_dt, dR_h_dt, dI_r_dt, dlambda_1_dt, dlambda_2_dt, dlambda_3_dt]

# Визначення системи диференціальних рівнянь без контролю
def model_no_control(t, y):
    I_h, R_h, I_r = y
    u = 0  # без контролю
    dI_h_dt = gamma_h * (1 - u) * I_r * (1 - I_h - R_h) - I_h * (lambda_hI + r1)
    dR_h_dt = r1 * I_h - R_h * (lambda_hS + r2)
    dI_r_dt = mu_rI * I_r - lambda_rI * I_r + gamma_r * (1 - u) * I_r * (1 - I_r)
    return [dI_h_dt, dR_h_dt, dI_r_dt]

# Інтеграція системи диференціальних рівнянь з контролем
t_span = [0, 2000]
y0 = [I_h0, R_h0, I_r0, lambda_10, lambda_20, lambda_30]
sol = solve_ivp(model, t_span, y0, method='RK45', t_eval=np.linspace(0, 2000, 1000))

# Інтеграція системи диференціальних рівнянь без контролю
y0_no_control = [I_h0, R_h0, I_r0]
sol_no_control = solve_ivp(model_no_control, t_span, y0_no_control, method='RK45', t_eval=np.linspace(0, 2000, 1000))

# Результати
t = sol.t
I_h, R_h, I_r, lambda_1, lambda_2, lambda_3 = sol.y
I_h_nc, R_h_nc, I_r_nc = sol_no_control.y

# Обчислення u(t)
u = I_r * (-I_h * gamma_h * lambda_1 - I_r * gamma_r * lambda_3 - R_h * gamma_h * lambda_1 + gamma_h * lambda_1 + gamma_r * lambda_3) / (2 * B)
u = np.clip(u, 0, 0.5)  # обмеження значення u в межах [0, 1]

# Усунення коливань u(t) за допомогою ковзного середнього
window_size = 10
u_smooth = np.convolve(u, np.ones(window_size)/window_size, mode='valid')
t_smooth = np.convolve(t, np.ones(window_size)/window_size, mode='valid')

# Візуалізація
plt.figure(figsize=(14, 10))

# Графік інфікованих людей
plt.subplot(2, 2, 1)
plt.plot(t, I_h, label='Інфіковані люди з контролем')
plt.plot(t, I_h_nc, label='Інфіковані люди без контролю', linestyle='--')
plt.xlabel('Час')
plt.ylabel('Популяція')
plt.legend()
plt.title('Динаміка інфікованих людей (I_h)')
plt.grid(True)

# Графік вилікуваних людей
plt.subplot(2, 2, 2)
plt.plot(t, R_h, label='Одужалі люди з контролем')
plt.plot(t, R_h_nc, label='Одужалі люди без контролю', linestyle='--')
plt.xlabel('Час')
plt.ylabel('Популяція')
plt.legend()
plt.title('Динаміка одужалих людей (R_h)')
plt.grid(True)

# Графік інфікованих гризунів
plt.subplot(2, 2, 3)
plt.plot(t, I_r, label='Інфіковані гризуни з контролем')
plt.plot(t, I_r_nc, label='Інфіковані гризуни без контролю', linestyle='--')
plt.xlabel('Час')
plt.ylabel('Популяція')
plt.legend()
plt.title('Динаміка інфікованих гризунів (I_r)')
plt.grid(True)

# Графік контрольної змінної u(t)
plt.subplot(2, 2, 4)
plt.plot(t_smooth, u_smooth, label='Контрольна змінна (u(t))', color='red')
plt.xlabel('Час')
plt.ylabel('Зусилля контролю (u)')
plt.legend()
plt.title('Контрольна змінна з часом')
plt.grid(True)

# Додатковий простір між підпідсюжетами
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()
