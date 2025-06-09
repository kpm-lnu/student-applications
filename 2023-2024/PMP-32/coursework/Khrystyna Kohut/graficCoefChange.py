import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def model(y, t, λ, ρ, β, δ, p, c):
    U, I, V = y
    dUdt = λ - ρ * U - β * U * V
    dIdt = β * U * V - δ * I
    dVdt = p * I - c * V
    return [dUdt, dIdt, dVdt]

U0 = 1000  # Початкова кількість вразливих клітин
I0 = 1     # Початкова кількість інфікованих клітин
V0 = 1     # Початкова кількість вірусних часток
y0 = [U0, I0, V0]

ρ = 0.01
δ = 0.15
β = 2.5
p = 15
c = 5

t = np.linspace(0, 30, 300)

beta_values = [0.01, 1, 10, 50]

plt.figure(figsize=(10, 6))

for λ in beta_values:
    solution = odeint(model, y0, t, args=(λ, ρ, β, δ, p, c))
    U, I, V = solution.T

    plt.plot(t, U, label=f'λ = {λ} (Вразливі клітини)')
    plt.plot(t, I, label=f'λ = {λ} (Інфіковані клітини)')
    plt.plot(t, V, label=f'λ = {λ} (Вірусні частки)')

plt.xlabel('Час')
plt.ylabel('Кількість клітин')
plt.title('Вплив параметра λ на динаміку поширення вірусу')
plt.legend()
plt.grid(True)
plt.show()
