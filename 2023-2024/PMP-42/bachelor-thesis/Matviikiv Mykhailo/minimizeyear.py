from pprint import pprint

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def get_data(path):
    date = []
    infected = []
    death = []
    recovered = []
    with open(path, 'r') as f:
        for x in f.readlines():
            temp = x.split()
            if len(temp) != 5:
                continue
            date.append(temp[0])
            infected.append(int(temp[2]))
            death.append(int(temp[3]))
            recovered.append(int(temp[4]))
    return date, infected, death, recovered


N = 41.1e6
DB = "2021.txt"
date, inf, d, r = get_data(DB)
data = [date, inf, r]
E0 = 0  # Initial exposed individuals
I0 = inf[0]  # Initial number of infected individuals
R0 = r[0]  # Initial number of recovered or deceased individuals
year_params = [[0.8975, 30, 3.4377],
               [1.9485, 30, 30],
               [1.7742, 30, 0.6],
               [1.9656, 30, 11.7],
               [0.6625, 30, 6.35],
               [0.4311, 30, 30],
               [4.0412, 30, 30],
               [1.7475, 30, 12.19],
               [2.3479, 30, 25.57],
               [1.1956, 30, 30],
               [1.7607, 30, 4.007],
               [6.3020, 30, 30]]


def SEIR_model_brovchenko(N, E0, I0, R0, days, draw=False):
    global inf, r
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0] = N - I0 - E0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    flag = 0
    for i in range(1, days):
        dSdt = -year_params[flag][0] / year_params[flag][1] * (1 / N) * S[i - 1] * I[i - 1]
        dEdt = (year_params[flag][0] / year_params[flag][1]) * (1 / N) * S[i - 1] * I[i - 1] - (
                E[i - 1] / year_params[flag][2])
        dIdt = (E[i - 1] / year_params[flag][2]) - (I[i - 1] / year_params[flag][1])
        dRdt = I[i - 1] / year_params[flag][1]
        S[i] = S[i - 1] + dSdt
        E[i] = E[i - 1] + dEdt
        I[i] = I[i - 1] + dIdt
        R[i] = R[i - 1] + dRdt
        if i % 30 == 0:
            flag += 1
            I[i] = inf[i]
            E[i] = 0
            R[i] = r[i]
            S[i] = N - I[i] - E[i]
    return S, E, I, R


S, E, I, R = SEIR_model_brovchenko(N, E0, I0, R0, 360, draw=False)

# Створюємо сплайн-інтерполяцію
cs = CubicSpline([x for x in range(0, 360)], I)

# Створюємо нові точки для більш гладкого графіку
x_new = np.linspace(0, 360)
y_new = cs(x_new)

date, i, d, r = get_data(DB)
# plt.plot(S, label="Здорові")
# plt.plot(E, label="Інкубаційні")
plt.plot(x_new, y_new, label="Інфіковані(result)")
# plt.plot(R, label=f"Одужані(result)")
plt.plot(i[:360], label="Інфіковані(DB)")
# plt.plot(r, label=f"Одужані(DB)")
plt.xlabel("Дні")
plt.ylabel("Кількість людей")
title = "SEIR COVID-19 в Україні(2021)"
plt.title(title)
plt.legend()
plt.show()
