import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Глобальні параметри
beta = 0.3      # Швидкість передачі для невакцинованих
beta_v = 0.06   # Швидкість передачі для вакцинованих (80% ефективність)
sigma = 0.2     # Швидкість переходу з латентного стану до інфекційного
gamma = 0.1     # Швидкість одужання
nu = 0.005      # Швидкість вакцинації
I0 = 0.001      # Початкова частка інфікованих
V0 = 0.3        # Початкова частка вакцинованих

# Значення для аналізу
v0_values = [0.0, 0.2, 0.4, 0.6]
efficacy_values = [0.0, 0.5, 0.8, 0.95]
nu_values = [0.001, 0.005, 0.01, 0.02]

def seirv_model(t, y, beta, beta_v, sigma, gamma, nu):
    S, E, I, R, V = y
    
    dSdt = -beta * S * I - nu * S
    dEdt = beta * S * I + beta_v * V * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dVdt = nu * S - beta_v * V * I
    
    return [dSdt, dEdt, dIdt, dRdt, dVdt]

def run_seirv_model(beta, beta_v, sigma, gamma, nu, I0, V0, t_max=200, dt=0.1):
    # Initial conditions
    S0 = 1 - I0 - V0
    E0 = 0
    R0 = 0
    y0 = [S0, E0, I0, R0, V0]
    
    # Time points
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    # Solve the ODE system
    sol = solve_ivp(
        seirv_model, t_span, y0, args=(beta, beta_v, sigma, gamma, nu),
        method='RK45', t_eval=t_eval
    )
    
    return sol.t, sol.y

def compute_outputs(t, y):
    S, E, I, R, V = y
    
    # Maximum fraction of infectious individuals
    I_max = np.max(I)
    t_max = t[np.argmax(I)]
    
    # Final fraction of recovered individuals
    R_final = R[-1]
    
    # Fraction of vaccinated individuals who got infected
    V0 = V[0]
    V_final = V[-1]
    V_infected = V0 - V_final
    
    return {
        'I_max': I_max,
        't_max': t_max,
        'R_final': R_final,
        'V_infected': V_infected
    }

def plot_seirv_model(t, y, title=None, filename=None):
    S, E, I, R, V = y
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Сприйнятливі (S)', linewidth=2)
    plt.plot(t, E, label='Експоновані (E)', linewidth=2)
    plt.plot(t, I, label='Інфіковані (I)', linewidth=2)
    plt.plot(t, R, label='Одужалі (R)', linewidth=2)
    plt.plot(t, V, label='Вакциновані (V)', linewidth=2)
    plt.xlabel('Час (дні)', fontsize=12)
    plt.ylabel('Частка популяції', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    if title:
        plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

# Запуск базового сценарію
t, y = run_seirv_model(beta, beta_v, sigma, gamma, nu, I0, V0)
outputs = compute_outputs(t, y)
plot_seirv_model(t, y, title='Модель SEIRV з неповним імунітетом від вакцинації', 
                filename='seirv_base_scenario.png')

# Різні рівні початкової вакцинації
plt.figure(figsize=(10, 6))
for v0 in v0_values:
    t, y = run_seirv_model(beta, beta_v, sigma, gamma, nu, I0, v0)
    plt.plot(t, y[2], label=f'V₀ = {v0}', linewidth=2)

plt.xlabel('Час (дні)', fontsize=12)
plt.ylabel('Частка інфікованих (I)', fontsize=12)
plt.title('Вплив початкового рівня вакцинації на динаміку епідемії', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('seirv_v0_comparison.png', dpi=300, bbox_inches='tight')

# Різні рівні ефективності вакцини
plt.figure(figsize=(10, 6))
for efficacy in efficacy_values:
    beta_v_val = beta * (1 - efficacy)
    t, y = run_seirv_model(beta, beta_v_val, sigma, gamma, nu, I0, V0)
    plt.plot(t, y[2], label=f'Ефективність = {efficacy*100:.0f}%', linewidth=2)

plt.xlabel('Час (дні)', fontsize=12)
plt.ylabel('Частка інфікованих (I)', fontsize=12)
plt.title('Вплив ефективності вакцини на динаміку епідемії', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('seirv_efficacy_comparison.png', dpi=300, bbox_inches='tight')

# Різні швидкості вакцинації
plt.figure(figsize=(10, 6))
for nu_val in nu_values:
    t, y = run_seirv_model(beta, beta_v, sigma, gamma, nu_val, I0, V0)
    plt.plot(t, y[2], label=f'ν = {nu_val:.3f}', linewidth=2)

plt.xlabel('Час (дні)', fontsize=12)
plt.ylabel('Частка інфікованих (I)', fontsize=12)
plt.title('Вплив швидкості вакцинації на динаміку епідемії', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('seirv_nu_comparison.png', dpi=300, bbox_inches='tight')

# Розрахунок базового репродуктивного числа
def calculate_R0(beta, gamma, sigma):
    return beta * sigma / (gamma * (gamma + sigma))

# Розрахунок ефективного репродуктивного числа з урахуванням вакцинації
def calculate_Re(beta, beta_v, gamma, sigma, S, V):
    return (beta * S + beta_v * V) * sigma / (gamma * (gamma + sigma))

# Розрахунок R0 для базового сценарію
R0 = calculate_R0(beta, gamma, sigma)

# Розрахунок Re для різних рівнів вакцинації
v0_values = np.linspace(0, 0.9, 10)
Re_values = []
for v0 in v0_values:
    s0 = 1 - I0 - v0
    Re = calculate_Re(beta, beta_v, gamma, sigma, s0, v0)
    Re_values.append(Re)

plt.figure(figsize=(10, 6))
plt.plot(v0_values * 100, Re_values, 'o-', linewidth=2)
plt.axhline(y=1, color='r', linestyle='--', label='Поріг епідемії (Re = 1)')
plt.xlabel('Початковий рівень вакцинації (%)', fontsize=12)
plt.ylabel('Ефективне репродуктивне число (Re)', fontsize=12)
plt.title('Залежність ефективного репродуктивного числа від рівня вакцинації', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('seirv_Re_vs_vaccination.png', dpi=300, bbox_inches='tight')

# Таблиця результатів для різних сценаріїв
print('Базове репродуктивне число (R0):', R0)

print('\nРезультати для різних рівнів початкової вакцинації:')
print('V0\tI_max\tt_max\tR_final')
for v0 in v0_values:
    t, y = run_seirv_model(beta, beta_v, sigma, gamma, nu, I0, v0)
    outputs = compute_outputs(t, y)
    print(f'{v0:.1f}\t{outputs["I_max"]:.4f}\t{outputs["t_max"]:.1f}\t{outputs["R_final"]:.4f}')

print('\nРезультати для різних рівнів ефективності вакцини:')
print('Ефективність\tI_max\tt_max\tR_final')
for efficacy in efficacy_values:
    beta_v_val = beta * (1 - efficacy)
    t, y = run_seirv_model(beta, beta_v_val, sigma, gamma, nu, I0, V0)
    outputs = compute_outputs(t, y)
    print(f'{efficacy*100:.0f}%\t{outputs["I_max"]:.4f}\t{outputs["t_max"]:.1f}\t{outputs["R_final"]:.4f}')

print('\nРезультати для різних швидкостей вакцинації:')
print('nu\tI_max\tt_max\tR_final')
for nu_val in nu_values:
    t, y = run_seirv_model(beta, beta_v, sigma, gamma, nu_val, I0, V0)
    outputs = compute_outputs(t, y)
    print(f'{nu_val:.3f}\t{outputs["I_max"]:.4f}\t{outputs["t_max"]:.1f}\t{outputs["R_final"]:.4f}')