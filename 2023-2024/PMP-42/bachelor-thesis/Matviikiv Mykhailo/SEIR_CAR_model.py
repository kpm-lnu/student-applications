import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def SEIR_CAR_model(N, E0, I0, R0, repro, tinf, tinc, days, tcar, pasym, draw=True):
    S = np.zeros(days)
    E = np.zeros(days);E1 = np.zeros(days)
    E2 = np.zeros(days);I = np.zeros(days)
    I1 = np.zeros(days);I2 = np.zeros(days)
    R = np.zeros(days);CAR = np.zeros(days)
    S[0] = N - I0;E[0] = E0
    E1[0] = E0;E2[0] = E0
    I[0] = I0;I1[0] = I0
    I2[0] = I0;R[0] = R0
    CAR[0] = R0
    for i in range(1, days):
        dSdt = -repro / tinf * (1 / N) * S[i - 1] * I[i - 1]
        dEdt = (repro / tinf) * (1 / N) * S[i - 1] * I[i - 1] - (E[i - 1] / tinc)
        dE1dt = pasym * (repro / tinf) * (1 / N) * S[i - 1] * I[i - 1] - (E1[i - 1] / tinc)
        dE2dt = (1 - pasym) * (repro / tinf) * (1 / N) * S[i - 1] * I[i - 1] - (E2[i - 1] / tinc)
        dIdt = (E[i - 1] / tinc) - (I[i - 1] / tinf)
        dI1dt = (E1[i - 1] / tinc) - (I1[i - 1] / tinf)
        dI2dt = (E2[i - 1] / tinc) - (I2[i - 1] / tcar)
        dRdt = I1[i - 1] / tinf
        dCARdt = I2[i - 1] / tcar

        S[i] = S[i - 1] + dSdt
        E[i] = E[i - 1] + dEdt
        E1[i] = E1[i - 1] + dE1dt
        E2[i] = E2[i - 1] + dE2dt
        I[i] = I[i - 1] + dIdt
        I1[i] = I1[i - 1] + dI1dt
        I2[i] = I2[i - 1] + dI2dt
        R[i] = R[i - 1] + dRdt
        CAR[i] = CAR[i - 1] + dCARdt
    if draw:
        plt.plot(S, label="Здорові s")
        plt.plot(E1, label="Інкубаційні (Tinc=3)e1")
        plt.plot(E2, label="Інкубаційні (Tinc=3)e2")
        plt.plot(I1, label="Інфіковані i1")
        plt.plot(I2, label="Інфіковані i2")
        plt.plot(R, label=f"Одужані R")
        plt.plot(CAR, label=f"CAR")
        plt.xlabel("Дні")
        plt.ylabel("Кількість людей")
        plt.title("SEIR_CAR модель для COVID-19 в Україні")
        plt.legend()
        plt.show()
    return S, E, E1, E2, I, I1, I2, R, CAR


N = 41.1e6  # Загальне населення України
E0 = 0  # Початкова кількість інфікованих осіб
I0 = 100  # Початкова кількість інфікованих осіб
R0 = 0  # Початкова кількість одужалих або померлих осіб
repro = 2.25  # Коефіцієнт контакту або коефіцієнт передачі
tinf = 4.5  # Коефіцієнт одужання або обернене середнього терміну інфікованості
tinc = 3  # інкубаційний період
days = 365  # Кількість днів для моделювання
tcar = 2
pasym = 0.3  # 30% хворих і безсимптомних

S, E, E1, E2, I, I1, I2, R, CAR = SEIR_CAR_model(N, E0, I0, R0, repro, tinf, tinc, days, tcar, pasym)
