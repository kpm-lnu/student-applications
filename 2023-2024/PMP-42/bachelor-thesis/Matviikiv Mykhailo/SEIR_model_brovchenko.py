import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def SEIR_model_brovchenko(N, E0, I0, R0, repro, tinf, tinc, days, draw=True):
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0] = N - I0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    for i in range(1, days):
        dSdt = -repro / tinf * (1 / N) * S[i - 1] * I[i - 1]
        dEdt = (repro / tinf) * (1 / N) * S[i - 1] * I[i - 1] - (E[i - 1] / tinc)
        dIdt = (E[i - 1] / tinc) - (I[i - 1] / tinf)
        dRdt = I[i - 1] / tinf
        S[i] = S[i - 1] + dSdt
        E[i] = E[i - 1] + dEdt
        I[i] = I[i - 1] + dIdt
        R[i] = R[i - 1] + dRdt
    if draw:
        # plt.plot(S, label="Здорові")
        # plt.plot(E, label="Інкубаційні")
        plt.plot(I, label="Інфіковані")
        # plt.plot(R, label=f"Одужані")
        plt.xlabel("Дні")
        plt.ylabel("Кількість людей(*10млн)")
        plt.title("SEIR модель для COVID-19 в Україні")
        plt.legend()
        plt.show()
    return S, E, I, R


N = 41.1e6  # Total population of Ukraine
days = 365  # Number of days for modeling
E0 = 0  # Initial exposed individuals
I0 = 2648  # Initial number of infected individuals
R0 = 62  # Initial number of recovered or deceased individuals
repro = 2.25   # Contact rate or transmission coefficient
tinf = 4.5  # Recovery rate or inverse of the mean infectious period
tinc = 4  # Incubation period

S, E, I, R = SEIR_model_brovchenko(N, E0, I0, R0, repro, tinf, tinc, days)
