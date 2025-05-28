import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp

# Оголошення змінних
gamma_h, lambda_hI, r1, lambda_hS, r2, mu_rI, lambda_rI, gamma_r = sp.symbols('gamma_h lambda_hI r1 lambda_hS r2 mu_rI lambda_rI gamma_r')

# Вираз для I_r
I_r_expr = (gamma_r - lambda_rI + mu_rI) / gamma_r
I_r_simplified = sp.simplify(I_r_expr)

# Вираз для D
D_expr = (gamma_h * gamma_r * lambda_hS + gamma_h * gamma_r * r1 + gamma_h * gamma_r * r2 -
          gamma_h * lambda_hS * lambda_rI + gamma_h * lambda_hS * mu_rI -
          gamma_h * lambda_rI * r1 - gamma_h * lambda_rI * r2 + gamma_h * mu_rI * r1 +
          gamma_h * mu_rI * r2 + gamma_r * lambda_hI * lambda_hS + gamma_r * lambda_hI * r2 +
          gamma_r * lambda_hS * r1 + gamma_r * r1 * r2)

# Вираз для I_h
I_h_expr = (gamma_h * (lambda_hS + r2) * (gamma_r - lambda_rI + mu_rI)) / D_expr
I_h_simplified = sp.simplify(I_h_expr)

# Вираз для R_h
R_h_expr = (gamma_h * r1 * (gamma_r - lambda_rI + mu_rI)) / D_expr
R_h_simplified = sp.simplify(R_h_expr)

# Перетворення виразів у функції для обчислення
I_r_func = sp.lambdify((gamma_h, lambda_hI, r1, lambda_hS, r2, mu_rI, lambda_rI, gamma_r), I_r_simplified)
I_h_func = sp.lambdify((gamma_h, lambda_hI, r1, lambda_hS, r2, mu_rI, lambda_rI, gamma_r), I_h_simplified)
R_h_func = sp.lambdify((gamma_h, lambda_hI, r1, lambda_hS, r2, mu_rI, lambda_rI, gamma_r), R_h_simplified)

# Параметри системи
gamma_h_val = 0.04
lambda_hI_val = 0.025
r1_val = 0.005
lambda_hS_val = 0.0001
r2_val = 0.0025
mu_rI_val = 0.02
lambda_rI_val = 0.04
gamma_r_val = 0.05

# Обчислення стаціонарних значень
I_r_val = I_r_func(gamma_h_val, lambda_hI_val, r1_val, lambda_hS_val, r2_val, mu_rI_val, lambda_rI_val, gamma_r_val)
I_h_val = I_h_func(gamma_h_val, lambda_hI_val, r1_val, lambda_hS_val, r2_val, mu_rI_val, lambda_rI_val, gamma_r_val)
R_h_val = R_h_func(gamma_h_val, lambda_hI_val, r1_val, lambda_hS_val, r2_val, mu_rI_val, lambda_rI_val, gamma_r_val)

print(f'Stationary point E1: I_h = {I_h_val}, R_h = {R_h_val}, I_r = {I_r_val}')

# Диференціальні рівняння
def system(vars, t, gamma_h, lambda_hI, r1, lambda_hS, r2, mu_rI, lambda_rI, gamma_r):
    I_h, R_h, I_r = vars
    dI_h_dt = gamma_h * I_r * (1 - I_h - R_h) - I_h * (lambda_hI + r1)
    dR_h_dt = r1 * I_h - R_h * (lambda_hS + r2)
    dI_r_dt = mu_rI * I_r - lambda_rI * I_r + gamma_r * I_r * (1 - I_r)
    return [dI_h_dt, dR_h_dt, dI_r_dt]

# Початкові умови поблизу стаціонарної точки (I_h_val, R_h_val, I_r_val)
initial_conditions = [
    [I_h_val * 1.021, R_h_val * 1.011, I_r_val * 1.037],
    [I_h_val * 1.01, R_h_val * 1.01, I_r_val * 1.03],
    [I_h_val * 1.031, R_h_val * 1.011, I_r_val * 1.035],
    [I_h_val * 1.021, R_h_val * 1.01, I_r_val * 1.035]
]

# Часовий інтервал
t = np.linspace(0, 500, 20000)  # збільшений інтервал часу

# Інтегрування системи для кожної початкової умови
trajectories = []
for ic in initial_conditions:
    sol = odeint(system, ic, t, args=(gamma_h_val, lambda_hI_val, r1_val, lambda_hS_val, r2_val, mu_rI_val, lambda_rI_val, gamma_r_val))
    trajectories.append(sol)

# Побудова фазового портрету
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'k']

for sol, color in zip(trajectories, colors):
    I_h, R_h, I_r = sol.T
    ax.plot(I_h, R_h, I_r, color=color)
    # Add quiver for each segment
    for i in range(0, len(I_h) - 1, 100):  # зменшення частоти стрілок для кращого вигляду
        ax.quiver(I_h[i], R_h[i], I_r[i], 
                  I_h[i + 1] - I_h[i], R_h[i + 1] - R_h[i], I_r[i + 1] - I_r[i], 
                  color=color, arrow_length_ratio=50)

ax.set_xlabel('I_h')
ax.set_ylabel('R_h')
ax.set_zlabel('I_r')
ax.set_title('Фазовий портрет системи поблизу стаціонарної точки E1')
ax.view_init(elev=20, azim=-150)  # зміна кута огляду для кращого відображення
plt.show()
