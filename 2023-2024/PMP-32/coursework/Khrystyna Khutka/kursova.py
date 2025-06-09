from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

N = 1
E0,I0, R0 = 0.25, 0.05, 0
S0 = N - I0 - R0-E0
y0 = S0, E0, I0, R0

t = np.linspace(0, 1000, 1000)

#55+
beta, gamma, mu, alpha = 0.17, 0.00923, 0.007, 0.22
#18-34
beta_1, gamma_1, mu_1, alpha_1 = 0.17, 0.00923, 0.009, 0.5
#35-54
beta_2, gamma_2, mu_2, alpha_2 = 0.17, 0.00923, 0.008, 0.33

#55+
def corrected_model(y, t, beta, mu, alpha, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - mu * E 
    dIdt = -gamma * I + alpha*mu * E
    dRdt = gamma * I + (1-alpha)* mu * E
    return [dSdt, dEdt, dIdt, dRdt]

#18-34
def corrected_model_1(y, t, beta_1, mu_1, alpha_1, gamma_1):
    S, E, I, R = y
    dSdt = -beta_1 * S * I
    dEdt = beta_1 * S * I - mu_1 * E 
    dIdt = -gamma_1 * I + alpha_1*mu_1 * E
    dRdt = gamma_1 * I + (1-alpha_1)* mu_1 * E
    return [dSdt, dEdt, dIdt, dRdt]

#35-54
def corrected_model_2(y, t, beta_2, mu_2, alpha_2, gamma_2):
    S, E, I, R = y
    dSdt = -beta_2 * S * I
    dEdt = beta_2* S * I - mu_2 * E 
    dIdt = -gamma_2 * I + alpha_2*mu_2 * E
    dRdt = gamma_2 * I + (1-alpha_2)* mu_2 * E
    return [dSdt, dEdt, dIdt, dRdt]


corrected_sol = odeint(corrected_model, [S0, E0, I0, R0], t, args=(beta, mu, alpha, gamma))
corrected_sol_1 = odeint(corrected_model_1, [S0, E0, I0, R0], t, args=(beta_1, mu_1, alpha_1, gamma_1))
corrected_sol_2 = odeint(corrected_model_2, [S0, E0, I0, R0], t, args=(beta_2, mu_2, alpha_2, gamma_2))


fig, axes = plt.subplots(1, 3, figsize=(14, 6))


axes[0].plot(t, corrected_sol_1[:, 0], label='Вразливі до фейку')
axes[0].plot(t, corrected_sol_1[:, 1], label='Дізнались новину')
axes[0].plot(t, corrected_sol_1[:, 2], label='Поширюють новину',color='red')
axes[0].plot(t, corrected_sol_1[:, 3], label='Дізнались правду', color='green')
axes[0].set_title('18-34')
axes[0].set_xlabel('Час')
axes[0].set_ylabel('Частка людей')
axes[0].legend()
axes[0].grid()

axes[1].plot(t, corrected_sol_2[:, 0], label='Вразливі до фейку')
axes[1].plot(t, corrected_sol_2[:, 1], label='Дізнались новину')
axes[1].plot(t, corrected_sol_2[:, 2], label='Поширюють новину', color='red')
axes[1].plot(t, corrected_sol_2[:, 3], label='Дізнались правду', color='green')
axes[1].set_title('35-54')
axes[1].set_xlabel('Час')
axes[1].set_ylabel('Частка людей')
axes[1].legend()
axes[1].grid()

axes[2].plot(t, corrected_sol[:, 0], label='Вразливі до фейку')
axes[2].plot(t, corrected_sol[:, 1], label='Дізнались новину')
axes[2].plot(t, corrected_sol[:, 2], label='Поширюють новину',color='red')
axes[2].plot(t, corrected_sol[:, 3], label='Дізнались правду', color='green')
axes[2].set_title('55+')
axes[2].set_xlabel('Час')
axes[2].set_ylabel('Частка людей')
axes[2].legend()
axes[2].grid() 


plt.tight_layout()
plt.show()



