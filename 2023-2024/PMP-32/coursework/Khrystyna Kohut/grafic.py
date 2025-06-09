import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# λ, ρ, β, δ, p, c = 0.1, 0.01, 2.5, 0.15, 15, 5
# λ, ρ, β, δ, p, c = 5, 0.00005, 0.01, 0.15, 2, 10
# λ, ρ, β, δ, p, c = 0.1, 0.01, 2.5, 0.15, 3, 5
λ=5
ρ=0.00005 
β=0.005 
δ=0.15 
p=2 
c=10

U0, I0, V0 = 998, 1, 1
# U0 = δ*c / (p * β)
# I0 = (λ * β * p - ρ * δ*c) / (δ * β * p)
# V0 = (λ * β * p - ρ * δ*c) / (δ * β * c)
initial_conditions = [U0, I0, V0]

def ebola_model(t, y):
    U, I, V = y
    dUdt = λ - ρ * U - β * U * V
    dIdt = β * U * V - δ * I
    dVdt = p * I - c * V
    return [dUdt, dIdt, dVdt]

t_span = (0, 30)
t_eval = np.linspace(*t_span, 300)

solution = solve_ivp(ebola_model, t_span, initial_conditions, t_eval=t_eval, method='RK45')

plt.plot(solution.t, solution.y[0], label='Вразливі клітини', color='blue')
plt.plot(solution.t, solution.y[1], label='Інфіковані клітини', color='red')
plt.plot(solution.t, solution.y[2], label='Вірусні частки', color='green')
plt.xlabel('Час')
plt.ylabel('Кількість клітин')
plt.title('Скоригована симуляція поширення вірусу Ебола')
plt.legend()
plt.grid()
plt.show()
