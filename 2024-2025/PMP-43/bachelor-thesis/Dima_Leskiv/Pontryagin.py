"""
Оптимальний контроль SIR-моделі з вакцинацією (метод Понтрягіна)
Параметри: ротавірус, популяція N = 1000.
Щепити можна максимум 2 % сприйнятливих на добу.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# 1. ПАРАМЕТРИ МОДЕЛІ
# ------------------------
beta    = 2.0       # β  – частота контактів
delta   = 0.2       # δ  – швидкість одужання
mu      = 1e-4      # μ  – природна смертність
Lambda_ = 1e-4      # λ  – народжуваність

# ваги у функціоналі: J = ∫ ( a I + (b/2) u² ) dt
a = 1.0
b = 0.5

N_pop = 1000.0      # розмір популяції (приблизно сталий)
u_max = 0.5        # 2 % сприйнятливих на добу

# ------------------------
# 2. ЧАСОВІ НАЛАШТУВАННЯ
# ------------------------
T       = 30.0     # днів
N_steps = 30      # по 1 дню
dt      = T / N_steps
t_grid  = np.linspace(0, T, N_steps + 1)

# ------------------------
# 3. ПОЧАТКОВІ УМОВИ
# ------------------------
S0, I0, R0 = 990.0, 50.0, 0.0

# ------------------------
# 4. ПРЯМА СИСТЕМА
# ------------------------
def rhs_forward(S, I, R, u):
    dS = Lambda_ - beta * S * I / N_pop - mu * S - u * S
    dI = beta * S * I / N_pop - (delta + mu) * I
    dR = delta * I - mu * R + u * S
    return np.array([dS, dI, dR])

def rk4_forward(u):
    S = np.zeros_like(t_grid); I = np.zeros_like(t_grid); R = np.zeros_like(t_grid)
    S[0], I[0], R[0] = S0, I0, R0
    for k in range(N_steps):
        y  = np.array([S[k], I[k], R[k]])
        k1 = rhs_forward(*y, u[k])
        k2 = rhs_forward(*(y + 0.5*dt*k1), u[k])
        k3 = rhs_forward(*(y + 0.5*dt*k2), u[k])
        k4 = rhs_forward(*(y + dt*k3),      u[k+1])
        S[k+1], I[k+1], R[k+1] = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return S, I, R

# ------------------------
# 5. СПРЯЖЕНА СИСТЕМА
# ------------------------
def rhs_adj(lS, lI, lR, S, I, R, u):
    dlS =  lS*(beta*I/N_pop + mu + u) - lI*(beta*I/N_pop) - lR*u
    dlI = -a + lS*(beta*S/N_pop) - lI*(beta*S/N_pop - delta - mu) - lR*delta
    dlR =  lR*mu
    return np.array([dlS, dlI, dlR])

def rk4_backward(S, I, R, u):
    lS = np.zeros_like(t_grid); lI = np.zeros_like(t_grid); lR = np.zeros_like(t_grid)
    for k in range(N_steps, 0, -1):
        y  = np.array([lS[k], lI[k], lR[k]])
        k1 = rhs_adj(*y,  S[k], I[k], R[k], u[k])
        k2 = rhs_adj(*(y - 0.5*dt*k1), S[k-1], I[k-1], R[k-1], u[k-1])
        k3 = rhs_adj(*(y - 0.5*dt*k2), S[k-1], I[k-1], R[k-1], u[k-1])
        k4 = rhs_adj(*(y - dt*k3),     S[k-1], I[k-1], R[k-1], u[k-1])
        lS[k-1], lI[k-1], lR[k-1] = y - (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return lS, lI, lR

# ------------------------
# 6. ОНОВЛЕННЯ КЕРУВАННЯ
# ------------------------
def update_control(S, lS, lR):
    u = (S * (lS - lR)) / b
    # обрізання: 0 ≤ u ≤ u_max
    return np.clip(u, 0.0, u_max)

# ------------------------
# 7. ІТЕРАЦІЙНИЙ АЛГОРИТМ
# ------------------------
u = np.zeros_like(t_grid)
max_iter, tol = 50, 1e-4

for it in range(max_iter):
    S, I, R       = rk4_forward(u)
    lS, lI, lR    = rk4_backward(S, I, R, u)
    u_new         = update_control(S, lS, lR)

    if np.linalg.norm(u_new - u, ord=np.inf) < tol:
        print(f"Збіжність за {it} ітерацій, ∥Δu∥∞ = {np.linalg.norm(u_new-u, np.inf):.2e}")
        u = u_new
        break
    u = u_new

# ------------------------
# 8. ВІЗУАЛІЗАЦІЯ
# ------------------------
plt.figure(figsize=(9,5))
plt.plot(t_grid, S, label="S(t)")
plt.plot(t_grid, I, label="I(t)")
plt.plot(t_grid, R, label="R(t)")
plt.ylabel("Осіб"); plt.xlabel("Дні")
plt.title("Динаміка S-I-R з оптимальним щепленням (u ≤ 2 %/день)")
plt.legend(); plt.tight_layout()

plt.figure(figsize=(9,3))
plt.plot(t_grid, u, label="u(t)")
plt.ylabel("Частка щеплених/день")
plt.xlabel("Дні"); plt.ylim(-0.01, u_max*1.1)
plt.title("Оптимальна стратегія вакцинації")
plt.tight_layout(); plt.show()
