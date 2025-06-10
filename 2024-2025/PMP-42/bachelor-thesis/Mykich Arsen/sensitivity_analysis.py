import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import seaborn as sns

beta_base = 0.3
sigma_base = 0.2
gamma_base = 0.1
nu_base = 0.05
beta_v_base = 0.05

S0 = 0.9
E0 = 0.0
I0 = 0.01
R0 = 0.0
V0 = 0.09

def seirv_model(t, y, beta, sigma, gamma, nu, beta_v):
    S, E, I, R, V = y
    
    dSdt = -beta * S * I - nu * S
    dEdt = beta * S * I + beta_v * V * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dVdt = nu * S - beta_v * V * I
    
    return [dSdt, dEdt, dIdt, dRdt, dVdt]

def run_simulation(beta, sigma, gamma, nu, beta_v):
    y0 = [S0, E0, I0, R0, V0]
    
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 1000)
    
    sol = solve_ivp(
        lambda t, y: seirv_model(t, y, beta, sigma, gamma, nu, beta_v),
        t_span,
        y0,
        method='RK45',
        t_eval=t_eval
    )
    
    S = sol.y[0]
    E = sol.y[1]
    I = sol.y[2]
    R = sol.y[3]
    V = sol.y[4]
    t = sol.t
    
    I_max = np.max(I)
    t_max = t[np.argmax(I)]
    R_final = R[-1]
    V_infected = V0 - V[-1] + nu_base * np.trapz(S, t)
    
    return {
        't': t,
        'S': S,
        'E': E,
        'I': I,
        'R': R,
        'V': V,
        'I_max': I_max,
        't_max': t_max,
        'R_final': R_final,
        'V_infected': V_infected
    }

def calculate_sensitivity_coefficients():
    parameters = ['beta', 'sigma', 'gamma', 'nu', 'beta_v']
    base_values = [beta_base, sigma_base, gamma_base, nu_base, beta_v_base]
    
    metrics = ['I_max', 't_max', 'R_final', 'V_infected']
    
    delta = 0.01
    
    base_results = run_simulation(beta_base, sigma_base, gamma_base, nu_base, beta_v_base)
    
    sensitivity = np.zeros((len(parameters), len(metrics)))
    
    for i, (param, base_value) in enumerate(zip(parameters, base_values)):
        perturbed_value = base_value * (1 + delta)
        
        perturbed_params = base_values.copy()
        perturbed_params[i] = perturbed_value
        
        perturbed_results = run_simulation(*perturbed_params)
        
        for j, metric in enumerate(metrics):
            base_metric = base_results[metric]
            perturbed_metric = perturbed_results[metric]
            
            if base_metric != 0:
                sensitivity[i, j] = ((perturbed_metric - base_metric) / base_metric) / delta
            else:
                sensitivity[i, j] = 0
    
    sensitivity_df = pd.DataFrame(sensitivity, index=parameters, columns=metrics)
    
    return sensitivity_df

def plot_sensitivity_coefficients(sensitivity_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = sns.color_palette("viridis", len(sensitivity_df.index))
    
    for i, metric in enumerate(sensitivity_df.columns):
        ax = axes[i]
        
        sorted_data = sensitivity_df[metric].sort_values(ascending=False)
        
        bars = ax.bar(sorted_data.index, sorted_data.values, color=colors)
        
        for bar in bars:
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_pos = height + 0.05
            else:
                va = 'top'
                y_pos = height - 0.05
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{height:.2f}', ha='center', va=va, fontsize=9)
        
        metric_labels = {
            'I_max': 'Максимальна частка інфікованих',
            't_max': 'Час досягнення піку епідемії',
            'R_final': 'Загальна частка перехворілих',
            'V_infected': 'Частка вакцинованих, які інфікувалися'
        }
        
        ax.set_title(f'Коефіцієнти чутливості для {metric_labels[metric]}', fontsize=12)
        ax.set_ylabel('Нормалізований коефіцієнт чутливості', fontsize=10)
        ax.set_xlabel('Параметр', fontsize=10)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        y_max = max(1.0, max(abs(sorted_data.values)) * 1.2)
        ax.set_ylim(-y_max, y_max)
    
    param_descriptions = {
        'beta': 'швидкість передачі інфекції',
        'sigma': 'швидкість переходу з E в I',
        'gamma': 'швидкість одужання',
        'nu': 'швидкість вакцинації',
        'beta_v': 'швидкість інфікування вакцинованих'
    }
    
    description_text = "Параметри моделі:\n"
    for param, desc in param_descriptions.items():
        description_text += f"{param}: {desc}\n"
    
    fig.text(0.5, 0.01, description_text, ha='center', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(hspace=0.3)
    
    fig.suptitle('Аналіз чутливості параметрів SEIRV-моделі', fontsize=16, y=0.98)
    
    plt.savefig('sensitivity_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Sensitivity coefficients plot saved to 'sensitivity_coefficients.png'")
    
    return sensitivity_df

def generate_sensitivity_analysis():
    sensitivity_df = calculate_sensitivity_coefficients()
    
    plot_sensitivity_coefficients(sensitivity_df)
    
    print("\nSensitivity Coefficients:")
    print(sensitivity_df)
    
    return sensitivity_df

if __name__ == "__main__":
    generate_sensitivity_analysis()