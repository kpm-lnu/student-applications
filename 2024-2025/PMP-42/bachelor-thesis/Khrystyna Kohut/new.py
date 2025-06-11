import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

os.makedirs("figures", exist_ok=True)

λ, ρ, β, δ, p, c = 0.1, 0.01, 0.0005, 0.15, 15, 5
A, B, C1 = 10, 20, 250
U0, I0, V0 = 998, 1, 1
T = 100
dt = 0.1
t = np.arange(0, T + dt, dt)
N_steps = len(t)

def system_rhs(t, y, m_func):
    U, I, V = y
    m = m_func(t)
    dU = λ - ρ * U - (1 - m) * β * U * V
    dI = (1 - m) * β * U * V - δ * I
    dV = p * I - c * V
    return [dU, dI, dV]

def run_simulation_and_plot(m_vals_input, scenario_name, file_suffix, plot_m=True):
    print(f"\n--- Симуляція: {scenario_name} ---")

    m_func = lambda t_val: np.interp(t_val, t, m_vals_input)
    sol = solve_ivp(system_rhs, [0, T], [U0, I0, V0], t_eval=t, args=(m_func,))
    U, I, V = sol.y
    J = np.trapz(A * I + B * V + 0.5 * C1 * m_vals_input**2, t)
    print(f"Функціонал якості J(m) = {J:.4f}")

    plt.figure(figsize=(12, 7))
    plt.plot(t, U, label="U (вразливі)", lw=2)
    plt.plot(t, I, label="I (інфіковані)", lw=2)
    plt.plot(t, V, label="V (віруси)", lw=2)
    plt.xlabel("Час (дні)"); plt.ylabel("Кількість")
    plt.title(f"Динаміка U(t), I(t), V(t) ({scenario_name})")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(f"figures/trajectories_{file_suffix}.png", dpi=300)
    plt.close()


    if plot_m:
        plt.figure(figsize=(12, 5))
        
        time_points = [0, 10, 60, T] 
        m_segment_values = [0.95, 0.7, 0.4]

        for i in range(len(m_segment_values)):
            start_time = time_points[i]
            end_time = time_points[i+1] if i + 1 < len(time_points) else T
            m_val = m_segment_values[i]
            plt.plot([start_time, end_time], [m_val, m_val], lw=2, color='C0') 

        plt.xlabel("Час (дні)"); plt.ylabel("m(t)")
        plt.title(f"Керування $m(t)$ ({scenario_name})")
        plt.ylim(-0.1, 1.1)
        plt.grid(); plt.tight_layout()
        plt.savefig(f"figures/m_control_{file_suffix}.png", dpi=300)
        plt.close()

    print(f"Графіки для '{scenario_name}' збережено у папці 'figures'")


m_vals_no_control = np.zeros_like(t)
run_simulation_and_plot(m_vals_no_control, "Без керування (m=0)", "no_control", plot_m=False)

time_points = [0, 10,60, T] 
m_segment_values = [0.95, 0.7, 0.4] 
m_vals_piecewise = np.zeros_like(t)
for i in range(len(m_segment_values)):
    start_time = time_points[i]
    end_time = time_points[i+1] if i + 1 < len(time_points) else T + dt
    indices = (t >= start_time) & (t < end_time)
    m_vals_piecewise[indices] = m_segment_values[i]
if t[-1] >= time_points[-2]:
    m_vals_piecewise[t == T] = m_segment_values[-1]

run_simulation_and_plot(m_vals_piecewise, "З кусково-постійним керуванням", "piecewise_control", plot_m=True)

print("\nВсі симуляції завершено.")