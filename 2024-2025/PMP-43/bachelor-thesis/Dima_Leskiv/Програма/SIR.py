import numpy as np
import matplotlib.pyplot as plt

# Параметри моделі
beta = 2.0      # Коефіцієнт передачі інфекції
delta = 0.2     # Швидкість одужання
mu = 0.0001     # Природна смертність
lambda_ = 0.0001  # Коефіцієнт народжуваності

# Початкові умови
N = 1000     # Загальна кількість населення
I0 = 10        # Початкова кількість інфікованих
R0 = 0          # Початкова кількість одужалих
S0 = N - I0 - R0  # Початкова кількість сприйнятливих

# Часові параметри
days = 30
dt = 1.0
t = np.linspace(0, days, int(days/dt) + 1)

# Ініціалізація масивів
S = np.zeros(len(t))
I = np.zeros(len(t))
R = np.zeros(len(t))

# Початкові значення
S[0] = S0
I[0] = I0
R[0] = R0

# Метод Рунге-Кутта 4-го порядку
for i in range(len(t) - 1):
    def dSdt(s, i_): return lambda_ * N - beta * s * i_ / N - mu * s
    def dIdt(s, i_): return beta * s * i_ / N - delta * i_ - mu * i_
    def dRdt(i_, r): return delta * i_ - mu * r

    k1_S = dSdt(S[i], I[i]) * dt
    k1_I = dIdt(S[i], I[i]) * dt
    k1_R = dRdt(I[i], R[i]) * dt

    k2_S = dSdt(S[i] + 0.5 * k1_S, I[i] + 0.5 * k1_I) * dt
    k2_I = dIdt(S[i] + 0.5 * k1_S, I[i] + 0.5 * k1_I) * dt
    k2_R = dRdt(I[i] + 0.5 * k1_I, R[i] + 0.5 * k1_R) * dt

    k3_S = dSdt(S[i] + 0.5 * k2_S, I[i] + 0.5 * k2_I) * dt
    k3_I = dIdt(S[i] + 0.5 * k2_S, I[i] + 0.5 * k2_I) * dt
    k3_R = dRdt(I[i] + 0.5 * k2_I, R[i] + 0.5 * k2_R) * dt

    k4_S = dSdt(S[i] + k3_S, I[i] + k3_I) * dt
    k4_I = dIdt(S[i] + k3_S, I[i] + k3_I) * dt
    k4_R = dRdt(I[i] + k3_I, R[i] + k3_R) * dt

    S[i+1] = S[i] + (k1_S + 2*k2_S + 2*k3_S + k4_S) / 6
    I[i+1] = I[i] + (k1_I + 2*k2_I + 2*k3_I + k4_I) / 6
    R[i+1] = R[i] + (k1_R + 2*k2_R + 2*k3_R + k4_R) / 6

# Побудова графіків
plt.figure(figsize=(10,6))
plt.plot(t, S, label='Сприйнятливі (S)')
plt.plot(t, I, label='Інфіковані (I)')
plt.plot(t, R, label='Одужалі (R)')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість осіб')
plt.title('Динаміка поширення ротавірусу (SIR-модель)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sir_rotavirus.png', dpi=300)
plt.show()
