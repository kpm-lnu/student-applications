import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pycountry
import requests

matplotlib.use('TkAgg')


def SIR_model(country_name, I0, R0, beta, gamma, days, draw=True):
    url = f'https://api.api-ninjas.com/v1/country?name={country_name}'

    response = requests.request("GET", url, headers={'X-Api-Key': 'V2ywxnIVCZ1VUt7VKbMWaA==8FCmUah4wGYgzQe0'})
    if response.status_code != requests.codes.ok:
        print(response.text)
        return
    N = int(json.loads(response.text)[0]['population'] * 1000)
    S = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0] = N - I0
    I[0] = I0
    R[0] = R0
    for i in range(1, days):
        # if i in list(range(40, 54)):
        #     beta =1.45
        # Calculate the change in S, I, and R per day
        dSdt = -beta / gamma * (1 / N) * S[i - 1] * I[i - 1]
        dIdt = beta / gamma * (1 / N) * S[i - 1] * I[i - 1] - I[i - 1] / gamma
        dRdt = I[i - 1] / gamma
        S[i] = S[i - 1] + dSdt
        I[i] = I[i - 1] + dIdt
        R[i] = R[i - 1] + dRdt
        # beta = 2.25

    if draw:
        # Plot the results
        plt.plot(S, label="Здорові")
        plt.plot(I, label="Інфіковані")
        plt.plot(R, label=f"Одужані")
        plt.xlabel("Дні")
        plt.ylabel("Кількість людей")
        plt.title("SIR модель ")
        plt.legend()
        plt.show()
    return S, I, R


I0 = 100  # Початкова кількість інфікованих осіб
R0 = 0  # Початкова кількість одужалих або померлих осіб
beta = 2.25  # Коефіцієнт контакту або коефіцієнт передачі
gamma = 4.5  # Коефіцієнт одужання або обернене середнього терміну інфікованості
days = 80  # Кількість днів для моделювання

S, I, R = SIR_model('ukraine', I0, R0, beta, gamma, days)
