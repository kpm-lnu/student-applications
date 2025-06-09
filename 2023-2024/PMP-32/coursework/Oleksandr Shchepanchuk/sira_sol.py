import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


I0, A0, R0 = 0.01, 0.01, 0
S0 = 1 - I0 - A0 - R0  

beta_1 = 0.1  
beta_2 = 0.2  
gamma = 0.09  
delta = 0.03  
k = 10  

t = np.linspace(0, 160, 160)

def deriv(y, t, N, beta_1, beta_2, gamma, delta, k):
    S, I, A, R = y
    dSdt = -beta_1 * S * I * k - beta_2 * S * A * k
    dIdt = beta_1 * S * I * k - gamma * I
    dAdt = delta * R
    dRdt = gamma * I - delta * R + beta_2 * S * A * k
    return dSdt, dIdt, dAdt, dRdt

y0 = S0, I0, A0, R0

ret = odeint(deriv, y0, t, args=(0, beta_1, beta_2, gamma, delta, k))
S, I, A, R = ret.T

plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', alpha=0.7, lw=2, label='Вразливі')
plt.plot(t, I, 'r', alpha=0.7, lw=2, label='Інфіковані')
plt.plot(t, A, 'g', alpha=0.7, lw=2, label='Анти-поширювачі')
plt.plot(t, R, 'm', alpha=0.7, lw=2, label='Вилікувані')
plt.title('Симуляція моделі SIRA')
plt.xlabel('Час')
plt.ylabel('Кількість індивідів')
plt.legend()
plt.grid(True)
plt.show()



beta_1 = 0.1 
gamma = 0.09  
delta = 0.03  
k = 10  

I0 = 0.1
R0 = 0

A0_values = [0.01, 0.05, 0.1]
beta_2_values = [0.05, 0.1, 0.15]

t = np.linspace(0, 100, 160)

plt.figure(figsize=(10, 6))

for A0 in A0_values:
    for beta_2 in beta_2_values:
        S0 = 1 - I0 - A0 - R0  
        y0 = S0, I0, A0, R0 
        ret = odeint(deriv, y0, t, args=(1, beta_1, beta_2, gamma, delta, k))
        S, I, A, R = ret.T
        plt.plot(t, I, lw=2, label=f'A0={A0}, β2={beta_2}')

plt.title('Вплив різних значень A0 та β2 на інфіковану популяцію')
plt.xlabel('Час')
plt.ylabel('Частка інфікованої популяції')
plt.legend()
plt.grid(True)
plt.show()

beta_1 = 0.1  
gamma = 0.09  
delta = 0.03  
k = 10 

I0 = 0.1
R0 = 0

A0_values = [0.01, 0.05, 0.1]
beta_2_values = [0.05, 0.1, 0.15]

t = np.linspace(0, 100, 160)

def deriv(y, t, N, beta_1, beta_2, gamma, delta, k):
    S, I, A, R = y
    dSdt = -beta_1 * S * I * k - beta_2 * S * A * k
    dIdt = beta_1 * S * I * k - gamma * I
    dAdt = delta * R
    dRdt = gamma * I - delta * R + beta_2 * S * A * k
    return dSdt, dIdt, dAdt, dRdt

plt.figure(figsize=(10, 6))


