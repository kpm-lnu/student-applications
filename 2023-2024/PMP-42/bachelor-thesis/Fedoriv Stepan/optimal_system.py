import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Початкові значення
y1_0 = 700
y2_0 = 500
a = 0.06
b = 0.5
c = 0.03
d = 0.35
T = 9
delta_t = 1

time = np.arange(0, T, delta_t)
n_steps = len(time)

# Початкові здогадки для управлінь
u_init = np.random.uniform(0, 100, n_steps)
v_init = np.random.uniform(0, 100, n_steps)

# Функція для моделювання системи
def model(y1_0, y2_0, u, v, n_steps, delta_t):
    y1 = y1_0
    y2 = y2_0
    y1_vals = [y1]
    y2_vals = [y2]

    for i in range(1, n_steps):
        y1 = y1 + (-a * y1 - b * y2 + u[i]) * delta_t
        y2 = y2 + (-c * y1 - d * y2 + v[i]) * delta_t
        y1_vals.append(y1)
        y2_vals.append(y2)

    return np.array(y1_vals), np.array(y2_vals)

# Цільова функція для мінімізації
def objective(uv):
    u = uv[:n_steps]
    v = uv[n_steps:]
    y1_vals, y2_vals = model(y1_0, y2_0, u, v, n_steps, delta_t)

    # Вагові коефіцієнти
    weight_y1 = 0.1
    weight_y2 = 0.1
    weight_u = 1
    weight_v = 1

    return (weight_y1 * (y1_vals[-1] - 250)**2 +
            weight_y2 * (y2_vals[-1] - 50)**2 +
            weight_u * np.sum(u**2) +
            weight_v * np.sum(v**2))

# Обмеження на управління (повинні бути між 0 і 100)
bounds = [(0, 100) for _ in range(2 * n_steps)]

# Початкова здогадка для оптимізації
uv_init = np.concatenate([u_init, v_init])

# Оптимізація
result = minimize(objective, uv_init, bounds=bounds, method='L-BFGS-B')

# Отримання оптимальних управлінь
u_opt = np.round(result.x[:n_steps]).astype(int)
v_opt = np.round(result.x[n_steps:]).astype(int)

# Моделювання з оптимальними управліннями
y1_vals_opt, y2_vals_opt = model(y1_0, y2_0, u_opt, v_opt, n_steps, delta_t)

print("Оптимальні підкріплення для армії A:", u_opt)
print("Оптимальні підкріплення для армії B:", v_opt)

# Побудова графіків
plt.plot(time, y1_vals_opt, label='Армія A (з оптимальним управлінням)')
plt.plot(time, y2_vals_opt, label='Армія B (з оптимальним управлінням)')
plt.xlabel('Час')
plt.ylabel('Чисельність')
plt.title('Модель з оптимальним управлінням')
plt.legend()
plt.grid()
plt.show()
