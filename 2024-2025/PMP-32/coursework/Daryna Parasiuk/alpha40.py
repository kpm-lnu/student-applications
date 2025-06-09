import numpy as np
from pyDOE import lhs
from scipy.integrate import solve_ivp
from scipy.interpolate import RBFInterpolator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Фіксовані параметри
beta = 0.02
delta = 0.01
gamma = 0.1

# Кількість навчальних і тестових точок
n_train = 40
n_test = 40

# Генеруємо навчальні та тестові точки через LHS
alpha_train = lhs(1, samples=n_train)
alpha_test = lhs(1, samples=n_test)

# Масштабуємо в [0.3, 1.5]
alpha_train = alpha_train * (1.5 - 0.3) + 0.3
alpha_test = alpha_test * (1.5 - 0.3) + 0.3


# Функція моделі хижак-жертва
def predator_prey_system(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Функція для моделювання і знаходження max популяції жертв
def simulate_max_prey(alpha_values):
    results = []
    for alpha in alpha_values.flatten():
        sol = solve_ivp(
            fun=lambda t, z: predator_prey_system(t, z, alpha, beta, delta, gamma),
            t_span=(0, 100),
            y0=[10, 5],
            t_eval=np.linspace(0, 100, 1000)
        )
        results.append(np.max(sol.y[0]))
    return np.array(results)

# Отримуємо y-значення
y_train = simulate_max_prey(alpha_train)
y_test_actual = simulate_max_prey(alpha_test)

# Створюємо RBF-інтерполятор
rbf_model = RBFInterpolator(alpha_train, y_train)
# rbf_model = RBFInterpolator(alpha_train, y_train, kernel='gaussian', epsilon=0.05)
# rbf_model = RBFInterpolator(alpha_train, y_train, kernel='cubic')

# Прогноз для тестових точок
y_test_pred = rbf_model(alpha_test)

# Метрики
mae = mean_absolute_error(y_test_actual, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
r2 = r2_score(y_test_actual, y_test_pred)

# Вивід
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

# Візуалізація
plt.figure(figsize=(8, 5))
plt.scatter(alpha_train, y_train, color='red', label='Навчальні точки')
plt.scatter(alpha_test, y_test_pred, color='blue', label='Прогнозовані тестові точки')
plt.xlabel('Alpha')
plt.ylabel('Max Prey')
plt.title('RBF-інтерполяція для моделі хижак-жертва (40 точок)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
