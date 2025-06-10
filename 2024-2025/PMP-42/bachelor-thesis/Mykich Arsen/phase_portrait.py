import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.cm as cm

beta = 0.3      
sigma = 0.2     
gamma = 0.1     
nu = 0.05       
beta_v = 0.05   

def seirv_model(t, y, beta, sigma, gamma, nu, beta_v):
    S, E, I, R, V = y
    
    dSdt = -beta * S * I - nu * S
    dEdt = beta * S * I + beta_v * V * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dVdt = nu * S - beta_v * V * I
    
    return [dSdt, dEdt, dIdt, dRdt, dVdt]

def generate_phase_portrait():
    plt.figure(figsize=(10, 8))
    
    S0_values = np.linspace(0.1, 0.9, 5)
    I0_values = np.linspace(0.01, 0.1, 5)
    
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 1000)
    
    colors = cm.viridis(np.linspace(0, 1, len(S0_values) * len(I0_values)))
    color_idx = 0
    
    for S0 in S0_values:
        for I0 in I0_values:
            E0 = 0.0
            R0 = 0.0
            V0 = 1.0 - S0 - E0 - I0 - R0  
            
            y0 = [S0, E0, I0, R0, V0]
            
            sol = solve_ivp(
                lambda t, y: seirv_model(t, y, beta, sigma, gamma, nu, beta_v),
                t_span,
                y0,
                method='RK45',
                t_eval=t_eval
            )
            
            S = sol.y[0]
            I = sol.y[2]
            
            plt.plot(S, I, '-', color=colors[color_idx], linewidth=1.5, 
                     label=f'S0={S0:.2f}, I0={I0:.2f}')
            
            plt.plot(S0, I0, 'o', color=colors[color_idx], markersize=6)
            
            color_idx += 1
    
    S_grid = np.linspace(0, 1, 20)
    I_grid = np.linspace(0, 0.3, 20)
    S_mesh, I_mesh = np.meshgrid(S_grid, I_grid)
    
    E_avg = 0.05
    R_avg = 0.2
    V_avg = 0.2
    
    dSdt = -beta * S_mesh * I_mesh - nu * S_mesh
    dIdt = sigma * E_avg - gamma * I_mesh
    
    norm = np.sqrt(dSdt**2 + dIdt**2)
    norm[norm == 0] = 1
    
    dSdt = dSdt / norm
    dIdt = dIdt / norm
    
    plt.quiver(S_mesh, I_mesh, dSdt, dIdt, color='gray', alpha=0.3)
    plt.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='S-nullcline')
    
    I_nullcline = sigma * E_avg / gamma
    plt.axhline(y=I_nullcline, color='red', linestyle='--', alpha=0.7, label='I-nullcline')
    
    plt.plot(0, 0, 'ko', markersize=8, label='Disease-free equilibrium')
    
    S_endemic = gamma / beta  
    if 0 < S_endemic < 1:
        I_endemic = nu * (1 - S_endemic) / (beta * S_endemic + beta_v * (1 - S_endemic))
        plt.plot(S_endemic, I_endemic, 'k*', markersize=10, label='Endemic equilibrium')
    
    plt.xlabel('Susceptible (S)', fontsize=12)
    plt.ylabel('Infectious (I)', fontsize=12)
    plt.title('Phase Portrait of SEIRV Model in S-I Plane', fontsize=14)
    
    plt.xlim(0, 1)
    plt.ylim(0, 0.3)
    
    plt.grid(True, alpha=0.3)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    selected_indices = list(range(len(handles) - 4, len(handles)))  
    selected_indices.extend([0, 6, 12, 18, 24])  
    plt.legend([handles[i] for i in selected_indices], [labels[i] for i in selected_indices], 
               loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('seirv_phase_portrait.png', dpi=300)
    plt.close()
    
    print("Phase portrait generated and saved to 'seirv_phase_portrait.png'")

def seir_model(t, y, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def generate_phase_portrait_seir():
    plt.figure(figsize=(10, 8))
    S0_values = np.linspace(0.1, 0.9, 5)
    I0_values = np.linspace(0.01, 0.1, 5)
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 1000)
    colors = cm.viridis(np.linspace(0, 1, len(S0_values) * len(I0_values)))
    color_idx = 0
    for S0 in S0_values:
        for I0 in I0_values:
            E0 = 0.0
            R0 = 1.0 - S0 - E0 - I0
            y0 = [S0, E0, I0, R0]
            sol = solve_ivp(
                lambda t, y: seir_model(t, y, beta, sigma, gamma),
                t_span, y0, method='RK45', t_eval=t_eval
            )
            S = sol.y[0]
            I = sol.y[2]
            plt.plot(S, I, '-', color=colors[color_idx], linewidth=1.5)
            plt.plot(S0, I0, 'o', color=colors[color_idx], markersize=6)
            color_idx += 1

    S_grid = np.linspace(0, 1, 20)
    I_grid = np.linspace(0, 0.3, 20)
    S_mesh, I_mesh = np.meshgrid(S_grid, I_grid)
    E_avg = 0.05
    dSdt = -beta * S_mesh * I_mesh
    dIdt = sigma * E_avg - gamma * I_mesh
    norm = np.sqrt(dSdt**2 + dIdt**2)
    norm[norm == 0] = 1
    dSdt = dSdt / norm
    dIdt = dIdt / norm
    plt.quiver(S_mesh, I_mesh, dSdt, dIdt, color='gray', alpha=0.3)
    plt.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='S-nullcline')
    I_nullcline = sigma * E_avg / gamma
    plt.axhline(y=I_nullcline, color='red', linestyle='--', alpha=0.7, label='I-nullcline')
    plt.plot(0, 0, 'ko', markersize=8, label='Disease-free equilibrium')
    S_endemic = gamma / beta
    if 0 < S_endemic < 1:
        I_endemic = 0.0
        plt.plot(S_endemic, I_endemic, 'k*', markersize=10, label='Endemic equilibrium')
    plt.xlabel('Susceptible (S)', fontsize=12)
    plt.ylabel('Infectious (I)', fontsize=12)
    plt.title('Phase Portrait of SEIR Model in S-I Plane', fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 0.3)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('seir_phase_portrait.png', dpi=300)
    plt.close()
    print("Phase portrait generated and saved to 'seir_phase_portrait.png'")

def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def generate_phase_portrait_sir():
    plt.figure(figsize=(10, 8))
    S0_values = np.linspace(0.1, 0.9, 5)
    I0_values = np.linspace(0.01, 0.1, 5)
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 1000)
    colors = cm.viridis(np.linspace(0, 1, len(S0_values) * len(I0_values)))
    color_idx = 0
    for S0 in S0_values:
        for I0 in I0_values:
            R0 = 1.0 - S0 - I0
            y0 = [S0, I0, R0]
            sol = solve_ivp(
                lambda t, y: sir_model(t, y, beta, gamma),
                t_span, y0, method='RK45', t_eval=t_eval
            )
            S = sol.y[0]
            I = sol.y[1]
            plt.plot(S, I, '-', color=colors[color_idx], linewidth=1.5)
            plt.plot(S0, I0, 'o', color=colors[color_idx], markersize=6)
            color_idx += 1

    S_grid = np.linspace(0, 1, 20)
    I_grid = np.linspace(0, 0.3, 20)
    S_mesh, I_mesh = np.meshgrid(S_grid, I_grid)
    dSdt = -beta * S_mesh * I_mesh
    dIdt = beta * S_mesh * I_mesh - gamma * I_mesh
    norm = np.sqrt(dSdt**2 + dIdt**2)
    norm[norm == 0] = 1
    dSdt = dSdt / norm
    dIdt = dIdt / norm
    plt.quiver(S_mesh, I_mesh, dSdt, dIdt, color='gray', alpha=0.3)
    plt.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='S-nullcline')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='I-nullcline')
    plt.plot(0, 0, 'ko', markersize=8, label='Disease-free equilibrium')
    S_endemic = gamma / beta
    if 0 < S_endemic < 1:
        plt.plot(S_endemic, 0, 'k*', markersize=10, label='Endemic equilibrium')
    plt.xlabel('Susceptible (S)', fontsize=12)
    plt.ylabel('Infectious (I)', fontsize=12)
    plt.title('Phase Portrait of SIR Model in S-I Plane', fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 0.3)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('sir_phase_portrait.png', dpi=300)
    plt.close()
    print("Phase portrait generated and saved to 'sir_phase_portrait.png'")

if __name__ == "__main__":
    generate_phase_portrait_sir()
    generate_phase_portrait_seir()
    generate_phase_portrait()
