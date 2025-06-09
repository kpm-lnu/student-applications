import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pycountry
import requests

matplotlib.use('TkAgg')


def SIR_model(I0, alpha, beta, y, days, draw=True):
    S = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0] = 1 - I0
    I[0] = I0
    R[0] = 0
    for i in range(1, days):
        dSdt = -alpha * S[i - 1] * I[i - 1] + y * R[i - 1]
        dIdt = alpha * S[i - 1] * I[i - 1] - beta * I[i - 1]
        dRdt = beta * I[i - 1] - y * R[i - 1]

        S[i] = S[i - 1] + dSdt
        I[i] = I[i - 1] + dIdt
        R[i] = R[i - 1] + dRdt
    if draw:
        # Plot the results
        plt.plot(S, label="Здорові")
        plt.plot(I, label="Інфіковані")
        plt.plot(R, label=f"Одужані")
        plt.xlabel("Дні")
        plt.ylabel("Кількість людей(*10млн)")
        plt.title("SIR модель для COVID-19 в Україні")
        plt.legend()
        plt.show()
    return S, I, R


S, I, R = SIR_model(0.000001, 0.5, 0.05, 0.04, 400)
