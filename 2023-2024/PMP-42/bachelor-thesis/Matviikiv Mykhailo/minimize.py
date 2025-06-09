import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

N = 41.1e6
repro_limits = [0.0, 10.0]
tinf_limits = [0, 30]
tinc_limits = [0, 30]


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


days_start = 110
days_finish = 140
DB = "2021.txt"
date, i, d, r = get_data(DB)
data = [date, i, r]
E0 = i[days_start]  # Initial exposed individuals
I0 = i[days_start]  # Initial number of infected individuals
R0 = r[days_start]  # Initial number of recovered or deceased individuals


def SEIR_model_brovchenko(N, E0, I0, R0, repro, tinf, tinc, draw=False):
    # S = np.zeros(days_finish - days_start)
    # E = np.zeros(days_finish - days_start)
    # I = np.zeros(days_finish - days_start)
    # R = np.zeros(days_finish - days_start)
    days_to_predict = 75
    S = np.zeros(days_to_predict)
    E = np.zeros(days_to_predict)
    I = np.zeros(days_to_predict)
    R = np.zeros(days_to_predict)
    S[0] = N - I0 - E0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    for i in range(1, days_to_predict):
    # for i in range(1, days_finish - days_start):
        dSdt = -repro / tinf * (1 / N) * S[i - 1] * I[i - 1]
        dEdt = (repro / tinf) * (1 / N) * S[i - 1] * I[i - 1] - (E[i - 1] / tinc)
        dIdt = (E[i - 1] / tinc) - (I[i - 1] / tinf)
        dRdt = I[i - 1] / tinf
        S[i] = S[i - 1] + dSdt
        E[i] = E[i - 1] + dEdt
        I[i] = I[i - 1] + dIdt
        R[i] = R[i - 1] + dRdt

    if draw:
        date, i, d, r = get_data(DB)
        # plt.plot(S, label="Здорові")
        # plt.plot(E, label="Інкубаційні")
        plt.plot(I, label="Інфіковані(result)")
        # plt.plot(R, label=f"Одужані(result)")
        plt.plot(i[days_start:days_finish+45], label="Інфіковані(DB)")
        # plt.plot(r[days_start:days_finish+45], label=f"Одужані(DB)")
        plt.xlabel("Дні")
        plt.ylabel("Кількість людей")
        title = "SEIR COVID-19 в Україні(2021)"
        plt.title(title)
        print(title)
        plt.legend()
        plt.show()

    return S, E, I, R


def objective_function(params):
    repro, tinf, tinc = params
    global N, E0, I0, R0

    S, E, I, R = SEIR_model_brovchenko(N, E0, I0, R0, repro, tinf, tinc)

    summ = 0
    for i in range(0, days_finish - days_start):
        summ += (((I[i] - data[1][i + days_start]) ** 2) + ((R[i] - data[2][i + days_start]) ** 2))

    return summ / (days_finish - days_start)


def constraint1(params):
    repro, tinf, tinc = params
    return repro - repro_limits[0]  # repro >= 1.0


def constraint2(params):
    repro, tinf, tinc = params
    return repro_limits[1] - repro  # repro <= 4.0


def constraint3(params):
    repro, tinf, tinc = params
    return tinf - tinf_limits[0]  # tinf >= 4.0


def constraint4(params):
    repro, tinf, tinc = params
    return tinf_limits[1] - tinf  # tinf <= 14.0


def constraint5(params):
    repro, tinf, tinc = params
    return tinc - tinc_limits[0]  # tinc >= 2.0


def constraint6(params):
    repro, tinf, tinc = params
    return tinc_limits[1] - tinc  # tinc <= 14.0


x0 = np.array([2.25, 4.5, 4.0])

constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2},
               {'type': 'ineq', 'fun': constraint3},
               {'type': 'ineq', 'fun': constraint4},
               {'type': 'ineq', 'fun': constraint5},
               {'type': 'ineq', 'fun': constraint6}]

solution = minimize(objective_function, x0, method='trust-constr', constraints=constraints, options={'verbose': 1})
print('Результат:', solution.x)
print('Значення цільової функції:', solution.fun)
S, E, I, R = SEIR_model_brovchenko(N, E0, I0, R0, solution.x[0], solution.x[1], solution.x[2], draw=True)
