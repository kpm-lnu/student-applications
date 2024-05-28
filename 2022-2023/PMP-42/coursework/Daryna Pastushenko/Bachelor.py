import numpy as np
import matplotlib.pyplot as plt

# Задані параметри
D = 1.0
v = np.array([1.0, 1.0])
f = 1.0

# Розмірність простору
L = 1.0

# Розбиття простору на вузли
N = 5
h = L / N
#x = np.linspace(0.0, L, (N+1)**2)
#y = np.linspace(0.0, L, (N+1)**2)
x = np.linspace(0.0, L, (N)**2)
y = np.linspace(0.0, L, (N)**2)


# Побудова матриць жорсткості та вектора правої частини
A = np.zeros((((N)**2), ((N)**2)))
b = np.zeros((N)**2)

for i in range(1, N**2 - 1):
    for j in range(1, N**2 - 1):
        # Обчислення локальних внесків
        a_ij = D / h**2 - np.dot(v, np.array([0.5*(y[j+1]-y[j-1]), 0.5*(x[i-1]-x[i+1])])) / (2*h)
        a_ip1j = -D / (2*h**2) + v[0] / (2*h)
        a_im1j = -D / (2*h**2) - v[0] / (2*h)
        a_ijp1 = -D / (2*h**2) + v[1] / (2*h)
        a_ijm1 = -D / (2*h**2) - v[1] / (2*h)
        b_ij = f

        # Збільшення локальних внесків в глобальні матрицю та вектор
        A[i, j] += a_ij
        A[i+1, j] += a_ip1j
        A[i-1, j] += a_im1j
        A[i, j+1] += a_ijp1
        A[i, j-1] += a_ijm1
        b[i] += b_ij

# Введення граничних умов
A[0, 0] = 1.0
A[-1, -1] = 1.0

A[np.triu_indices(N**2, k=2)] = 0.0
A[np.tril_indices(N**2, k=-2)] = 0.0

b[0] = 0.0
b[-1] = 0.0

print(A)

# Розв'язок системи лінійних рівнянь
u = np.linalg.solve(A, b)

print(u)

# Візуалізація результату
X, Y = np.meshgrid(np.linspace(0.0, L, N), np.linspace(0.0, L, N))
plt.contourf(X, Y, u.reshape((N, N)), cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Розподіл концентрації речовини')
plt.show()