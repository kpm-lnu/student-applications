import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def SEIR_model(S0, E0, I0, R0, D0, ae, ai, b, y, k, u, p, days, draw=True):
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    D = np.zeros(days)
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    D[0] = D0
    for i in range(1, days):
        # temp = ae
        # if i in range(20,35):
        #     ae = 0.1
        dSdt = \
            -ae * S[i - 1] * E[i - 1] - ai * S[i - 1] * I[i - 1] + y * R[i - 1]
        dEdt = \
            ae * S[i - 1] * E[i - 1] + ai * S[i - 1] * I[i - 1] - k * E[i - 1] - p * E[i - 1]
        dIdt = k * E[i - 1] - b * I[i - 1] - u * I[i - 1]
        dRdt = b * I[i - 1] + p * E[i - 1] - y * R[i - 1]
        dDdt = u * I[i - 1]
        S[i] = S[i - 1] + dSdt
        E[i] = E[i - 1] + dEdt
        I[i] = I[i - 1] + dIdt
        R[i] = R[i - 1] + dRdt
        D[i] = D[i - 1] + dDdt
        # ae = temp

    if draw:
        print(S)
        plt.plot(S, label="Здорові")
        plt.plot(E, label="Інкубаційні")
        plt.plot(I, label="Інфіковані")
        plt.plot(R, label=f"Одужані")
        plt.plot(D, label=f"Померлі")
        plt.xlabel("Дні")
        plt.ylabel("Співвідношення людей")
        plt.title("SEIR модель для COVID-19")
        plt.legend()
        plt.show()
    return S, E, I, R, D


S, E, I, R, D = SEIR_model(S0=0.9998, E0=0.0001, I0=0.0001, R0=0, D0=0,
                           ae=0.65, ai=0.005, k=0.05, p=0.08,
                           b=0.1, u=0.02, y=0.01, days=400)


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


date, i, d, r = get_data("2020.txt")
data = [date, i, d]

