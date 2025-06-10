import numpy as np
import matplotlib.pyplot as plt

# Параметри моделі
lambda_ = 0.0001   # народжуваність
beta = 0.6       # зменшено для стабільності
mu = 0.0001        # природна смертність
delta = 0.2        # швидкість одужання
theta = 0.2        # частка вакцинованих при народженні
eta = 0.05         # швидкість вакцинації
zeta = 0.01        # швидкість втрати імунітету
xi = 0.1           # ризик зараження вакцинованих
tau = 0.01         # смертність від хвороби

# Початкові умови
N = 1000
S0 = 950
V0 = 0
I0 = 50
R0 = 0

# Часові параметри
days = 30
dt = 0.1  # зменшено крок для стабільності
t = np.linspace(0, days, int(days/dt) + 1)

# Масиви для розв'язків
S = np.zeros(len(t))
V = np.zeros(len(t))
I = np.zeros(len(t))
R = np.zeros(len(t))

S[0], V[0], I[0], R[0] = S0, V0, I0, R0

# Означення диференціальних рівнянь
def dSdt(s, v, i):
    return (1 - theta)*lambda_*N - beta*s*i/N + zeta*v - eta*s - mu*s

def dVdt(s, v, i):
    return theta*lambda_*N + eta*s - xi*beta*v*i/N - (zeta + mu)*v

def dIdt(s, v, i):
    return beta*s*i/N + xi*beta*v*i/N - (tau + delta + mu)*i

def dRdt(i, r):
    return delta*i - mu*r

# Інтегрування методом Рунге-Кутта 4-го порядку
for j in range(len(t)-1):
    k1_S = dSdt(S[j], V[j], I[j])*dt
    k1_V = dVdt(S[j], V[j], I[j])*dt
    k1_I = dIdt(S[j], V[j], I[j])*dt
    k1_R = dRdt(I[j], R[j])*dt

    k2_S = dSdt(S[j]+0.5*k1_S, V[j]+0.5*k1_V, I[j]+0.5*k1_I)*dt
    k2_V = dVdt(S[j]+0.5*k1_S, V[j]+0.5*k1_V, I[j]+0.5*k1_I)*dt
    k2_I = dIdt(S[j]+0.5*k1_S, V[j]+0.5*k1_V, I[j]+0.5*k1_I)*dt
    k2_R = dRdt(I[j]+0.5*k1_I, R[j]+0.5*k1_R)*dt

    k3_S = dSdt(S[j]+0.5*k2_S, V[j]+0.5*k2_V, I[j]+0.5*k2_I)*dt
    k3_V = dVdt(S[j]+0.5*k2_S, V[j]+0.5*k2_V, I[j]+0.5*k2_I)*dt
    k3_I = dIdt(S[j]+0.5*k2_S, V[j]+0.5*k2_V, I[j]+0.5*k2_I)*dt
    k3_R = dRdt(I[j]+0.5*k2_I, R[j]+0.5*k2_R)*dt

    k4_S = dSdt(S[j]+k3_S, V[j]+k3_V, I[j]+k3_I)*dt
    k4_V = dVdt(S[j]+k3_S, V[j]+k3_V, I[j]+k3_I)*dt
    k4_I = dIdt(S[j]+k3_S, V[j]+k3_V, I[j]+k3_I)*dt
    k4_R = dRdt(I[j]+k3_I, R[j]+k3_R)*dt

    S[j+1] = S[j] + (k1_S + 2*k2_S + 2*k3_S + k4_S)/6
    V[j+1] = V[j] + (k1_V + 2*k2_V + 2*k3_V + k4_V)/6
    I[j+1] = I[j] + (k1_I + 2*k2_I + 2*k3_I + k4_I)/6
    R[j+1] = R[j] + (k1_R + 2*k2_R + 2*k3_R + k4_R)/6

    # Захист від переповнення
    S[j+1] = max(min(S[j+1], N), 0)
    V[j+1] = max(min(V[j+1], N), 0)
    I[j+1] = max(min(I[j+1], N), 0)
    R[j+1] = max(min(R[j+1], N), 0)

# Графіки результатів
plt.figure(figsize=(10,6))
plt.plot(t, S, label='S (сприйнятливі)')
plt.plot(t, V, label='V (вакциновані)')
plt.plot(t, I, label='I (інфіковані)')
plt.plot(t, R, label='R (одужалі)')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість осіб')
plt.title('Динаміка поширення ротавірусу з вакцинацією (SIRV-модель)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sirv_rotavirus_stable.png', dpi=300)
plt.show()
