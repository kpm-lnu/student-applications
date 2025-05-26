import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
from matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.font_manager import FontProperties

alpha1, beta1, delta1, sigma1 = 0.7, 0.3, 1, 0.5
alpha3, beta2, gamma2, sigma2 = 0.98, 0.7, 0.8, 0.4
s, rho, omega, gamma3, mu, sigma3, v = 0.4, 0.2, 0.3, 0.3, 0.29, 0.5, 1
Pi, theta = 0.5, 1

B = 0.01

def system(t, y, u):
    H, T, I, E = y
    dH_dt = H * (alpha1 - beta1 * H - delta1 * T) - (1 - u) * sigma1 * H * E
    dT_dt = T * (alpha3 - beta2 * T) - gamma2 * I * T + (1 - u) * sigma2 * H * E
    dI_dt = s + (rho * I * T) / (omega + T) - gamma3 * I * T - mu * I - (1 - u) * (sigma3 * I * E) / (v + E)
    dE_dt = (1 - u) * Pi - theta * E
    return [dH_dt, dT_dt, dI_dt, dE_dt]

y0 = [1, 0.00001, 1.379310345, 0.5]
t_span = (0, 30)
t_eval = np.linspace(t_span[0], t_span[1], 32)

def adjoint_system(t, lambda_, x, u):
    x1, x2, x3, x4 = x
    lambda1, lambda2, lambda3, lambda4 = lambda_

    dlambda1 = lambda1 * (2 * beta1 * x1 - alpha1 + delta1 * x2 + (1 - u) * sigma1 * x4) - lambda2 * (
                (1 - u) * sigma2 * x4)
    dlambda2 = lambda1 * delta1 * x1 - lambda2 * (alpha3 - 2 * beta2 * x2 - gamma2 * x3) - lambda3 * (
                (rho * x3 * omega) / ((omega + x2) ** 2) - gamma3 * x3) - 1
    dlambda3 = lambda2 * gamma2 * x2 - lambda3 * (
                (rho * x2) / (omega + x2) - gamma3 * x2 - mu - (1 - u) * (sigma3 * x4) / (v + x4))
    dlambda4 = (1 - u) * (lambda1 * sigma1 * x1 - lambda2 * sigma2 * x1 + lambda3 * (sigma3 * x3 * v) / (
                (v + x4) ** 2)) + lambda4 * theta

    return [dlambda1, dlambda2, dlambda3, dlambda4]

lambda0 = [0, 0, 0, 0]

def update_control(u_old, y, lambda_, t_eval, alpha=0.01, threshold=0.7):
    u_new = np.zeros_like(u_old)
    for i, t in enumerate(t_eval):
        H, T, I, E = y[:, i]
        lambda_H, lambda_T, lambda_I, lambda_E = lambda_[:, i]
        # Розрахунок градієнта функції Гамільтона
        grad_Hamil = lambda_H * (-sigma1 * H * E) + lambda_T * (-sigma2 * H * E) + lambda_I * (-sigma3 * I * E / (v + E)) + lambda_E * Pi + B * u_old[i]
        # Оновлення контролю з врахуванням градієнта функції Гамільтона
        u_new[i] = min(max(-grad_Hamil / B, 0), 1)
        # Додаткова умова: якщо нормальні клітини зменшуються більш ніж на 30%, вводимо препарат
        if H < threshold * y0[0]:
            u_new[i] = 1
    return alpha * u_new + (1 - alpha) * u_old

def solve_optimal_control():
    u = np.concatenate((np.zeros(len(t_eval) // 2), np.ones(len(t_eval) // 2)))
    tol = 1e-3
    max_iter = 100

    for _ in range(max_iter):
        sol = solve_ivp(lambda t, y: system(t, y, u[int(t / t_span[1] * (len(t_eval) - 1))]), t_span, y0, t_eval=t_eval)
        y = sol.y

        lambda_sol = solve_ivp(lambda t, lambda_: adjoint_system(t, lambda_, y[:, int((t_span[1] - t) / t_span[1] * (len(t_eval) - 1))], u[int((t_span[1] - t) / t_span[1] * (len(t_eval) - 1))]), (t_span[1], t_span[0]), lambda0, t_eval=t_eval[::-1])
        lambda_ = lambda_sol.y[:, ::-1]

        u_new = update_control(u, y, lambda_, t_eval)

        if np.linalg.norm(u_new - u) < tol:
            break
        u = u_new

    return y, u

def calculate_functional(t_eval, y, u, B):
    T = y[1, :]
    J = np.trapz(T + 0.5 * B * u**2, t_eval)
    return J

y, u = solve_optimal_control()
H, T, I, E = y

J = calculate_functional(t_eval, y, u, B)
print(f"Значення функціоналу: {J}")

plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})

plt.plot(t_eval, H, label='Нормальні клітини (H)', color='blue', linewidth=3)
plt.plot(t_eval, T, label='Пухлинні клітини (T)', color='red',linewidth=3)
plt.plot(t_eval, I, label='Імунні клітини (I)', color='green',linewidth=3)
plt.xlabel('Час',fontdict={'fontsize': 16, 'weight': 'bold'})
plt.ylabel('Популяції',fontdict={'fontsize': 16, 'weight': 'bold'})
legend_properties = FontProperties(size=16, weight='bold')
plt.legend((r'$H(t)$', r'$T(t)$', r'$I(t)$'), loc='upper right', bbox_to_anchor=(1.1, 1), prop=legend_properties)
plt.title('Динаміка клітинних популяцій з оптимальним контролем',fontdict={'fontsize': 16, 'weight': 'bold'})

plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})

plt.hlines(u[:-1], t_eval[:-1], t_eval[1:], colors='black', label='Керуюча змінна (u)', linewidth=3)
plt.xlabel('Час',fontdict={'fontsize': 16, 'weight': 'bold'})
plt.ylabel('Сила боротьби',fontdict={'fontsize': 16, 'weight': 'bold'})
plt.title('Оптимальна функція контролю за часом',fontdict={'fontsize': 16, 'weight': 'bold'})
plt.legend()
plt.grid(True)
plt.show()